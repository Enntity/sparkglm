# SPDX-License-Identifier: Apache-2.0
# Adapted from NvmeDirectOffloadingSpec in
# https://github.com/gabewillen/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
# at 3c2add4c491737c5b916217e0c5d6dc708fe02d8 (PR #232).
"""Experimental GPU-to-NVMe prefix-cache offload for SparkGLM.

The implementation keeps the upstream direct-file model: each one-block chunk
is written atomically, with no resident CPU cache tier and no eviction thread.
SparkGLM's experimental additions are a required restart namespace, private
0700 directories and 0600 files, SHA-256 verification before GPU copy, and a
per-store free-space check.

Scheduler lookups use durable commit records published only after successful
store acknowledgements from every TP rank. Each rank's payload stays on its
local SSD. Failed loads invalidate the commit and trigger a fully fenced cold
recompute through the accompanying connector patch. No live file sweeper is
allowed. This remains an opt-in experiment pending workload qualification.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import shutil
import threading
from threading import local as thread_local
from pathlib import Path
from collections import deque
from collections.abc import Collection
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait

import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    TransferResult,
)
from vllm.v1.kv_offload.config import OffloadingConfig

logger = init_logger(__name__)

_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
_DIGEST_BYTES = hashlib.sha256(b"").digest_size


def _write_all(fd: int, view: memoryview) -> None:
    """Write exactly the supplied bytes, tolerating short OS writes."""
    mv = memoryview(view).cast("B")
    offset = 0
    while offset < len(mv):
        written = os.write(fd, mv[offset:])
        if written <= 0:
            raise OSError(f"write returned {written} at offset {offset}/{len(mv)}")
        offset += written


def _read_all(fd: int, view: memoryview) -> int:
    """Fill the buffer; the return is bytes read (< length only at EOF)."""
    mv = memoryview(view).cast("B")
    offset = 0
    while offset < len(mv):
        read = os.readv(fd, [mv[offset:]])
        if read == 0:
            break
        offset += read
    return offset


def _key_relpath(key: OffloadKey) -> str:
    """Stable relative path for an offload key (hash and group bytes)."""
    digest = bytes(key).hex()
    return os.path.join(digest[:3], f"{digest}.bin")


def _private_dir(path: str) -> str:
    os.makedirs(path, mode=0o700, exist_ok=True)
    # umask can mask the intended mode when the directory already exists.
    os.chmod(path, 0o700)
    return path


def _read_payload(path: str, expected: int):
    """Validate the entire encoded chunk before exposing any bytes to CUDA."""
    encoded = bytearray(expected + _DIGEST_BYTES)
    fd = os.open(path, os.O_RDONLY)
    try:
        if os.fstat(fd).st_size != len(encoded):
            raise OSError(f"wrong cache file size on {path}")
        got = _read_all(fd, memoryview(encoded))
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (OSError, AttributeError):
            pass
    finally:
        os.close(fd)
    if got != len(encoded):
        raise OSError(f"short read on {path}")
    payload = memoryview(encoded)[:expected]
    if not hmac.compare_digest(hashlib.sha256(payload).digest(), bytes(encoded[expected:])):
        raise OSError(f"checksum mismatch on {path}")
    return payload


class NvmeFileLoadStoreSpec(LoadStoreSpec):
    """Relative file names, ordered like the job's keys and GPU blocks."""

    def __init__(self, relpaths: list[str]):
        self.relpaths = relpaths

    def __repr__(self) -> str:
        return f"NvmeFileLoadStoreSpec({len(self.relpaths)} files)"


class NvmeDirectManager(OffloadingManager):
    """Rank-zero commit index; publish only after all TP writers acknowledge."""

    def __init__(self, root_dir: str, namespace: str, world_size: int):
        self.medium = Medium.STORAGE
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self._namespace_dir = _private_dir(os.path.join(root_dir, namespace))
        self._commits = _private_dir(os.path.join(self._namespace_dir, "committed"))
        self._pending_stores: set[OffloadKey] = set()
        self._invalid: set[OffloadKey] = set()

    def _path(self, key):
        return os.path.join(self._commits, _key_relpath(key) + ".ready")

    def on_new_request(self, req_context):
        return RequestOffloadingContext()

    def lookup(self, key, req_context):
        if key in self._invalid:
            return LookupResult.MISS
        if key in self._pending_stores:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT if os.path.isfile(self._path(key)) else LookupResult.MISS

    def prepare_load(self, keys, req_context):
        return NvmeFileLoadStoreSpec([_key_relpath(k) for k in keys])

    def prepare_store(self, keys, req_context):
        selected = [k for k in keys if k not in self._pending_stores
                    and self.lookup(k, req_context) is LookupResult.MISS]
        self._pending_stores.update(selected)
        return PrepareStoreOutput(selected,
            NvmeFileLoadStoreSpec([_key_relpath(k) for k in selected]), [])

    def complete_store(self, keys, req_context, success=True):
        self._pending_stores.difference_update(keys)
        if not success:
            self.invalidate(keys)
            return
        for key in keys:
            path = self._path(key)
            try:
                _private_dir(os.path.dirname(path))
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                _sync_dir(os.path.dirname(path))
                self._invalid.discard(key)
            except OSError:
                self.invalidate([key])
                logger.warning("NVMe commit record failed; treating key as a miss")

    def invalidate(self, keys):
        for key in keys:
            self._invalid.add(key)
            try:
                os.unlink(self._path(key))
                _sync_dir(os.path.dirname(self._path(key)))
            except FileNotFoundError:
                pass

    def reset_cache(self):
        self._pending_stores.clear()
        # Keep both durable commits and in-process invalidations.


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class NvmeDirectWorker(OffloadingWorker):
    """Worker-side GPU<->NVMe streamer with per-thread pinned bounce buffers."""

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        rank: int,
        root_dir: str,
        namespace: str,
        capacity_bytes: int,
        reserve_bytes: int,
        n_io_threads: int = 4,
    ):
        self._namespace_dir = _private_dir(os.path.join(root_dir, namespace))
        self._dir = _private_dir(os.path.join(self._namespace_dir, f"r{rank}"))
        self._capacity_bytes = int(capacity_bytes)
        self._reserve_bytes = int(reserve_bytes)
        self._required_free_bytes = self._reserve_bytes
        self._capacity_lock = threading.Lock()
        self._used_bytes = sum(p.stat().st_size for p in Path(self._dir).rglob('*') if p.is_file())
        self._tensors = [tensor.tensor for tensor in kv_caches.tensors]
        self._group_refs = kv_caches.group_data_refs

        self._max_chunk_bytes = max(
            (
                sum(reference.page_size_bytes for reference in references)
                for references in self._group_refs
                if references
            ),
            default=0,
        )
        self._tls = thread_local()
        self._pool = ThreadPoolExecutor(
            max_workers=n_io_threads, thread_name_prefix="vllm_kv_nvme"
        )
        self._jobs: dict[int, tuple[Future, int, bool]] = {}
        self._results: deque[TransferResult] = deque()
        logger.info(
            "NvmeDirectWorker rank=%d dir=%s tensors=%d threads=%d "
            "bounce=%.1f MiB/thread (max %.1f MiB total)",
            rank,
            self._dir,
            len(self._tensors),
            n_io_threads,
            self._max_chunk_bytes / (1 << 20),
            n_io_threads * self._max_chunk_bytes / (1 << 20),
        )

    def _thread_state(self):
        state = getattr(self._tls, "state", None)
        if state is None:
            buffer = torch.empty(
                self._max_chunk_bytes, dtype=torch.uint8, pin_memory=True
            )
            state = (buffer, buffer.numpy(), torch.cuda.Stream())
            self._tls.state = state
        return state

    def _plan(self, gpu_spec: GPULoadStoreSpec, relpaths: list[str]):
        """Return chunk plans and bytes; blocks_per_chunk must be one."""
        block_ids = gpu_spec.block_ids
        assert len(relpaths) == len(block_ids), (
            f"{len(relpaths)} files vs {len(block_ids)} blocks"
        )
        plans = []
        total = 0
        index = 0
        for group_idx, group_size in enumerate(gpu_spec.group_sizes):
            if group_size == 0:
                continue
            group_bytes = sum(
                reference.page_size_bytes for reference in self._group_refs[group_idx]
            )
            for block_id in block_ids[index : index + group_size]:
                plans.append(
                    (os.path.join(self._dir, relpaths[index]), group_idx, int(block_id))
                )
                total += group_bytes
                index += 1
        assert index == len(block_ids)
        return plans, total

    def _check_capacity(self, required_bytes: int) -> None:
        """Reserve quota before IO; failed stores retain a conservative charge."""
        if required_bytes <= 0:
            return
        with self._capacity_lock:
            usage = shutil.disk_usage(self._namespace_dir)
            if usage.free - required_bytes < self._required_free_bytes:
                raise OSError("NVMe prefix-cache free-space reserve reached")
            if self._capacity_bytes and self._used_bytes + required_bytes > self._capacity_bytes:
                raise OSError("NVMe prefix-cache capacity reached")
            self._used_bytes += required_bytes

    def _store_task(self, event: torch.cuda.Event, plans) -> None:
        buffer, buffer_np, stream = self._thread_state()
        stream.wait_event(event)
        for path, group_idx, block_id in plans:
            expected = sum(
                reference.page_size_bytes
                for reference in self._group_refs[group_idx]
            )
            self._check_capacity(expected + _DIGEST_BYTES)
            offset = 0
            with torch.cuda.stream(stream):
                for reference in self._group_refs[group_idx]:
                    size = reference.page_size_bytes
                    buffer[offset : offset + size].copy_(
                        self._tensors[reference.tensor_idx][block_id, :size].view(
                            torch.uint8
                        ),
                        non_blocking=True,
                    )
                    offset += size
            stream.synchronize()
            assert offset == expected

            payload = bytes(buffer_np[:offset].data)
            digest = hashlib.sha256(payload).digest()
            output = memoryview(payload + digest)
            os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
            os.chmod(os.path.dirname(path), 0o700)
            tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.fchmod(fd, 0o600)
                _write_all(fd, output)
                os.fsync(fd)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
            except BaseException:
                os.close(fd)
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            os.close(fd)
            os.replace(tmp, path)
            _sync_dir(os.path.dirname(path))

    def _load_task(self, event: torch.cuda.Event, plans) -> None:
        buffer, buffer_np, stream = self._thread_state()
        stream.wait_event(event)
        for path, group_idx, block_id in plans:
            expected = sum(
                reference.page_size_bytes
                for reference in self._group_refs[group_idx]
            )
            payload = _read_payload(path, expected)

            # The checksum is verified before any byte is copied to the GPU.
            buffer_np[:expected] = payload
            offset = 0
            with torch.cuda.stream(stream):
                for reference in self._group_refs[group_idx]:
                    size = reference.page_size_bytes
                    self._tensors[reference.tensor_idx][block_id, :size].view(
                        torch.uint8
                    ).copy_(buffer[offset : offset + size], non_blocking=True)
                    offset += size
            stream.synchronize()

    def _submit(
        self,
        job_id: int,
        gpu_spec: GPULoadStoreSpec,
        relpaths: list[str],
        is_load: bool,
    ) -> bool:
        plans, total = self._plan(gpu_spec, relpaths)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        task = self._load_task if is_load else self._store_task
        future = self._pool.submit(task, event, plans)
        self._jobs[job_id] = (future, total, is_load)
        return True

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(dst_spec, NvmeFileLoadStoreSpec)
        return self._submit(job_id, src_spec, dst_spec.relpaths, is_load=False)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        assert isinstance(src_spec, NvmeFileLoadStoreSpec)
        return self._submit(job_id, dst_spec, src_spec.relpaths, is_load=True)

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        finished = [job_id for job_id, (future, _, _) in self._jobs.items() if future.done()]
        for job_id in finished:
            future, total, is_load = self._jobs.pop(job_id)
            exception = future.exception()
            if exception is not None:
                if is_load:
                    logger.error("NVMe KV load job %d failed: %s", job_id, exception)
                logger.warning(
                    "NVMe KV store job %d failed (block stays un-offloaded): %s",
                    job_id,
                    exception,
                )
            results.append(
                TransferResult(
                    job_id=job_id,
                    success=exception is None,
                    transfer_size=total,
                    transfer_time=None,
                )
            )
        return results

    def wait(self, job_ids: set[int]) -> None:
        futures = [self._jobs[job_id][0] for job_id in job_ids if job_id in self._jobs]
        if futures:
            futures_wait(futures)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


class NvmeDirectOffloadingSpec(OffloadingSpec):
    """Strictly opt-in wiring for the experimental NVMe prefix cache."""

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        for name in ("root_dir", "namespace", "capacity_bytes", "reserve_bytes"):
            if name not in self.extra_config:
                raise ValueError(
                    f"{name} is required in kv_connector_extra_config for "
                    "NvmeDirectOffloadingSpec"
                )

        self.root_dir = str(self.extra_config["root_dir"])
        self.namespace = str(self.extra_config["namespace"])
        if not self.root_dir:
            raise ValueError("root_dir must be a non-empty string")
        if not _NAMESPACE_RE.fullmatch(self.namespace):
            raise ValueError(
                "namespace must pin the checkpoint/build/layout identity and use "
                "8-128 safe characters [A-Za-z0-9._-]"
            )
        self.capacity_bytes = int(self.extra_config["capacity_bytes"])
        self.reserve_bytes = int(self.extra_config["reserve_bytes"])
        if self.capacity_bytes <= 0 or self.reserve_bytes <= 0:
            raise ValueError("capacity_bytes and reserve_bytes must be positive")
        self.n_io_threads = int(self.extra_config.get("n_io_threads", 4))
        if self.n_io_threads <= 0:
            raise ValueError("n_io_threads must be positive")

        _private_dir(self.root_dir)
        namespace_dir = os.path.join(self.root_dir, self.namespace)
        _private_dir(namespace_dir)
        usage = shutil.disk_usage(namespace_dir)
        required_free = max(self.capacity_bytes, self.reserve_bytes)
        if usage.free < required_free:
            raise RuntimeError(
                f"NVMe prefix-cache target {namespace_dir} has {usage.free} bytes "
                f"free; {required_free} bytes required"
            )
        if self.blocks_per_chunk != 1:
            raise ValueError(
                "NvmeDirectOffloadingSpec requires blocks_per_chunk == 1"
            )
        logger.info(
            "NVMe prefix-cache startup validated: namespace=%s free=%.1f GiB "
            "required=%.1f GiB; no eviction thread",
            self.namespace,
            usage.free / (1 << 30),
            required_free / (1 << 30),
        )
        self._manager: NvmeDirectManager | None = None
        self._worker: NvmeDirectWorker | None = None

    @override
    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            self._manager = NvmeDirectManager(
                root_dir=self.root_dir,
                namespace=self.namespace,
                world_size=self.config.parallel.world_size,
            )
            logger.info(
                "Created committed-rank NvmeDirectManager for %s (world=%d)",
                os.path.join(self.root_dir, self.namespace),
                self.config.parallel.world_size - 1,
            )
        return self._manager

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            self._worker = NvmeDirectWorker(
                kv_caches=kv_caches,
                rank=self.config.parallel.rank,
                root_dir=self.root_dir,
                namespace=self.namespace,
                capacity_bytes=self.capacity_bytes,
                reserve_bytes=self.reserve_bytes,
                n_io_threads=self.n_io_threads,
            )
        return self._worker
