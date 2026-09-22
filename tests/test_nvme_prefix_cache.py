# SPDX-License-Identifier: Apache-2.0
"""Bounded stdlib-only tests for the NVMe prefix-cache product module.

The product module imports torch and vllm eagerly, so these tests AST-extract
the pure helpers (``_write_all``, ``_read_all``, ``_key_relpath``,
``_private_dir``, ``_sync_dir``) and load ``NvmeDirectManager`` against
minimal ``sys.modules`` stubs. Everything runs without torch/vllm installed.
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / 'overlay/kvoffload/nvme_direct.py'

_PURE_NAMES = (
    "_write_all",
    "_read_all",
    "_key_relpath",
    "_private_dir",
    "_sync_dir",
)


def _extract_pure_helpers():
    """Compile only the standalone helper functions from the product module."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _PURE_NAMES
    ]
    missing = set(_PURE_NAMES) - {node.name for node in wanted}
    if missing:
        raise AssertionError(f"helpers not found in module: {sorted(missing)}")
    module = ast.Module(body=wanted, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {"os": os, "memoryview": memoryview, "OffloadKey": bytes}
    exec(compile(module, str(MODULE_PATH), "exec"), namespace)  # noqa: S102
    ns = types.SimpleNamespace(**namespace)

    # ``os``/``memoryview`` are resolved as module globals at call time, so
    # swapping ``ns.os`` is enough to inject a fake FD layer.
    def _write_all(fd, view):
        namespace["os"] = ns.os
        return namespace["_write_all"](fd, view)

    def _read_all(fd, view):
        namespace["os"] = ns.os
        return namespace["_read_all"](fd, view)

    def _sync_dir(path):
        namespace["os"] = ns.os
        return namespace["_sync_dir"](path)

    ns.write_all = _write_all
    ns.read_all = _read_all
    ns.sync_dir = _sync_dir
    ns.key_relpath = namespace["_key_relpath"]
    ns.private_dir = namespace["_private_dir"]
    return ns


def _install_stubs():
    """Install the minimum torch/vllm surface NvmeDirectManager touches."""

    class LoadStoreSpec:
        pass

    class OffloadingManager:
        pass

    class OffloadingSpec:
        pass

    class OffloadingWorker:
        pass

    class Medium:
        STORAGE = "STORAGE"

    class LookupResult:
        HIT = "HIT"
        MISS = "MISS"
        HIT_PENDING = "HIT_PENDING"

    class RequestOffloadingContext:
        pass

    class CanonicalKVCaches:
        pass

    class GPULoadStoreSpec(LoadStoreSpec):
        pass

    class PrepareStoreOutput:
        def __init__(self, keys, store_spec, evicted_keys):
            self.keys = keys
            self.store_spec = store_spec
            self.evicted_keys = evicted_keys

    class TransferResult:
        def __init__(self, job_id, success, transfer_size, transfer_time):
            self.job_id = job_id
            self.success = success
            self.transfer_size = transfer_size
            self.transfer_time = transfer_time

    class ReqContext:
        pass

    class OffloadKey(bytes):
        pass

    torch = types.ModuleType("torch")
    torch.uint8 = "uint8"
    torch.empty = lambda *a, **k: None
    torch.cuda = types.SimpleNamespace(Event=object, Stream=object)

    vllm = types.ModuleType("vllm")
    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.init_logger = lambda name: types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    base = types.ModuleType("vllm.v1.kv_offload.base")
    base.CanonicalKVCaches = CanonicalKVCaches
    base.GPULoadStoreSpec = GPULoadStoreSpec
    base.LoadStoreSpec = LoadStoreSpec
    base.LookupResult = LookupResult
    base.Medium = Medium
    base.OffloadingManager = OffloadingManager
    base.OffloadingSpec = OffloadingSpec
    base.OffloadingWorker = OffloadingWorker
    base.OffloadKey = OffloadKey
    base.PrepareStoreOutput = PrepareStoreOutput
    base.ReqContext = ReqContext
    base.RequestOffloadingContext = RequestOffloadingContext
    base.TransferResult = TransferResult
    config = types.ModuleType("vllm.v1.kv_offload.config")
    config.OffloadingConfig = object

    stubs = {
        "torch": torch,
        "typing_extensions": types.ModuleType("typing_extensions"),
        "vllm": vllm,
        "vllm.logger": vllm_logger,
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.kv_offload": types.ModuleType("vllm.v1.kv_offload"),
        "vllm.v1.kv_offload.base": base,
        "vllm.v1.kv_offload.config": config,
    }
    for name, module in stubs.items():
        if name == "typing_extensions":
            module.override = lambda fn: fn
        sys.modules[name] = module
    sys.modules["vllm.v1"].kv_offload = sys.modules["vllm.v1.kv_offload"]
    sys.modules["vllm.v1.kv_offload"].base = base
    return LookupResult


def _load_manager_module():
    expected_lookup = _install_stubs()
    spec = importlib.util.spec_from_file_location(
        "nvme_direct_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, expected_lookup


class ShortIOTests(unittest.TestCase):
    """``_write_all``/``_read_all`` must survive partial OS transfers."""

    def setUp(self):
        self.ns = _extract_pure_helpers()

    def test_write_all_loops_over_short_writes(self):
        class FakeOs:
            def __init__(self, chunk):
                self.chunk = chunk
                self.calls = 0
                self.sink = bytearray()

            def write(self, fd, view):
                mv = memoryview(view).cast("B")
                take = min(self.chunk, len(mv))
                if take <= 0:
                    raise AssertionError("empty write requested")
                self.sink += bytes(mv[:take])
                self.calls += 1
                return take

        payload = bytes(range(256)) * 4
        fake = FakeOs(chunk=7)
        ns = self.ns
        ns.os = fake
        ns.write_all(3, memoryview(payload))
        self.assertEqual(bytes(fake.sink), payload)
        self.assertGreater(fake.calls, 1)

        # A zero-length write must be rejected, not looped forever.
        ns.os = types.SimpleNamespace(write=lambda fd, view: 0)
        with self.assertRaises(OSError):
            ns.write_all(3, memoryview(b"abc"))

    def _fake_reader(self, chunk, data):
        state = {"pos": 0, "data": bytearray(data)}

        def readv(fd, buffers):
            mv = memoryview(buffers[0]).cast("B")
            remaining = len(state["data"]) - state["pos"]
            take = max(0, min(chunk, len(mv), remaining))
            if take == 0:
                return 0
            mv[:take] = state["data"][state["pos"] : state["pos"] + take]
            state["pos"] += take
            return take

        return types.SimpleNamespace(readv=readv), state

    def test_read_all_reports_partial_bytes_at_eof(self):
        payload = bytes(range(200))
        fake, state = self._fake_reader(chunk=13, data=payload)
        self.ns.os = fake
        buffer = bytearray(len(payload))
        got = self.ns.read_all(3, memoryview(buffer))
        self.assertEqual(got, len(payload))
        self.assertEqual(bytes(buffer), payload)

        short, _ = self._fake_reader(chunk=5, data=payload[:40])
        self.ns.os = short
        truncated = bytearray(200)
        self.assertEqual(self.ns.read_all(3, memoryview(truncated)), 40)

    def test_key_relpath_is_sharded_and_stable(self):
        key = bytes.fromhex("ab" * 32)
        expected = os.path.join(key.hex()[:3], f"{key.hex()}.bin")
        self.assertEqual(self.ns.key_relpath(key), expected)
        self.assertEqual(self.ns.key_relpath(key), self.ns.key_relpath(key))
        self.assertEqual(len(expected.split(os.sep)), 2)

    def test_private_dir_forces_0700(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "ns")
            os.makedirs(target, mode=0o755)
            os.chmod(target, 0o755)
            self.ns.private_dir(target)
            self.assertEqual(os.stat(target).st_mode & 0o777, 0o700)
            self.assertTrue(os.path.isdir(target))

    def test_sync_dir_opens_and_fsyncs_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            opened = {}

            class FakeOs:
                O_RDONLY = os.O_RDONLY
                O_DIRECTORY = os.O_DIRECTORY

                def open(self, path, flags):
                    opened["path"] = path
                    opened["flags"] = flags
                    return 9

                def fsync(self, fd):
                    opened["fsync"] = fd

                def close(self, fd):
                    opened["closed"] = fd

            self.ns.os = FakeOs()
            self.ns.sync_dir(tmp)
            self.assertEqual(opened["path"], tmp)
            self.assertTrue(opened["flags"] & os.O_DIRECTORY)
            self.assertEqual(opened["fsync"], 9)
            self.assertEqual(opened["closed"], 9)


class ManagerTests(unittest.TestCase):
    """Pending -> all-rank-complete success -> durable HIT across recreation."""

    @classmethod
    def setUpClass(cls):
        cls.module, cls.LookupResult = _load_manager_module()
        if cls.module.LookupResult is not cls.LookupResult:
            raise AssertionError("stubs not wired into loaded module")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.key = b"\x07" * 32
        self.other = b"\x08" * 32

    def tearDown(self):
        self._tmp.cleanup()

    def _manager(self, namespace="model.build.layout"):
        return self.module.NvmeDirectManager(
            root_dir=self.root, namespace=namespace, world_size=2
        )

    def test_zero_world_size_rejected(self):
        with self.assertRaises(ValueError):
            self.module.NvmeDirectManager(
                root_dir=self.root, namespace="ns-00000001", world_size=0
            )

    def test_pending_then_completed_keys_are_hits_across_recreation(self):
        mgr = self._manager()
        ctx = mgr.on_new_request(None)
        LR = self.LookupResult

        self.assertIs(mgr.lookup(self.key, ctx), LR.MISS)

        out = mgr.prepare_store([self.key, self.other], ctx)
        self.assertEqual(list(out.keys), [self.key, self.other])
        self.assertEqual(len(out.store_spec.relpaths), 2)
        self.assertIs(mgr.lookup(self.key, ctx), LR.HIT_PENDING)

        # Reprepare must not re-select a pending key.
        again = mgr.prepare_store([self.key], ctx)
        self.assertEqual(list(again.keys), [])

        # The scheduler calls complete_store only after aggregating all ranks.
        # Here we test publication for two keys, not rank aggregation itself.
        mgr.complete_store([self.key], ctx, success=True)
        self.assertIs(mgr.lookup(self.key, ctx), LR.HIT)
        self.assertIs(mgr.lookup(self.other, ctx), LR.HIT_PENDING)
        mgr.complete_store([self.other], ctx, success=True)
        self.assertIs(mgr.lookup(self.other, ctx), LR.HIT)

        ready = os.path.join(self.root, "model.build.layout", "committed")
        committed = [
            os.path.join(dirpath, name)
            for dirpath, _, names in os.walk(ready)
            for name in names
            if name.endswith(".ready")
        ]
        self.assertEqual(len(committed), 2)
        for path in committed:
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            # Commit records are grouped under a 3-hex-char shard directory.
            self.assertEqual(
                os.path.relpath(os.path.dirname(path), ready).count(os.sep) + 1, 1
            )
            self.assertEqual(len(os.path.basename(os.path.dirname(path))), 3)
        self.assertEqual(os.stat(ready).st_mode & 0o777, 0o700)

        # A fresh manager over the same root must observe durable commits.
        recreated = self._manager()
        ctx2 = recreated.on_new_request(None)
        self.assertIs(recreated.lookup(self.key, ctx2), LR.HIT)
        self.assertIs(recreated.lookup(self.other, ctx2), LR.HIT)

        load_spec = recreated.prepare_load([self.key], ctx2)
        self.assertEqual(load_spec.relpaths, [self.module._key_relpath(self.key)])
        self.assertIn("1 files", repr(load_spec))

    def test_failed_complete_store_misses_and_removes_commit(self):
        mgr = self._manager()
        ctx = mgr.on_new_request(None)
        LR = self.LookupResult
        mgr.complete_store([self.key], ctx, success=True)
        self.assertIs(mgr.lookup(self.key, ctx), LR.HIT)
        path = mgr._path(self.key)
        self.assertTrue(os.path.isfile(path))

        mgr.prepare_store([self.key], ctx)
        mgr.complete_store([self.key], ctx, success=False)
        self.assertIs(mgr.lookup(self.key, ctx), LR.MISS)
        self.assertFalse(os.path.exists(path))

        # Recreating does not resurrect the failed commit.
        self.assertIs(
            self._manager().lookup(self.key, mgr.on_new_request(None)), LR.MISS
        )

    def test_invalidate_key_overrides_disk_ready(self):
        mgr = self._manager()
        ctx = mgr.on_new_request(None)
        LR = self.LookupResult
        mgr.prepare_store([self.key], ctx)
        mgr.complete_store([self.key], ctx, success=True)
        self.assertIs(mgr.lookup(self.key, ctx), LR.HIT)

        mgr.invalidate([self.key])
        self.assertIs(mgr.lookup(self.key, ctx), LR.MISS)
        self.assertIn(self.key, mgr._invalid)
        self.assertFalse(os.path.exists(mgr._path(self.key)))

        # Store is revoked, but the commit index stays durable. A later cycle
        # may only publish again after a fresh all-rank-complete success.
        mgr.complete_store([self.key], ctx, success=True)
        self.assertIs(mgr.lookup(self.key, ctx), LR.HIT)
        self.assertNotIn(self.key, mgr._invalid)

        # A different key is in flight when invalidated.
        mgr.prepare_store([self.other], ctx)
        self.assertIs(mgr.lookup(self.other, ctx), LR.HIT_PENDING)
        mgr.invalidate([self.other])
        self.assertIs(mgr.lookup(self.other, ctx), LR.MISS)

    def test_reset_cache_clears_pending_but_keeps_commits_and_invalid(self):
        mgr = self._manager()
        ctx = mgr.on_new_request(None)
        LR = self.LookupResult
        mgr.prepare_store([self.key, self.other], ctx)
        mgr.invalidate([self.other])
        mgr.reset_cache()
        self.assertEqual(mgr._pending_stores, set())
        self.assertIn(self.other, mgr._invalid)
        self.assertIs(mgr.lookup(self.other, ctx), LR.MISS)

    def test_prepare_store_skips_already_hit_keys(self):
        mgr = self._manager()
        ctx = mgr.on_new_request(None)
        mgr.prepare_store([self.key], ctx)
        mgr.complete_store([self.key], ctx, success=True)
        out = mgr.prepare_store([self.key], ctx)
        self.assertEqual(list(out.keys), [])
        self.assertEqual(out.store_spec.relpaths, [])

    def test_namespace_validation_and_private_layout(self):
        self._manager(namespace="Model-1.2_build")
        self.assertTrue(
            os.path.isdir(os.path.join(self.root, "Model-1.2_build", "committed"))
        )
        self.assertEqual(
            os.stat(os.path.join(self.root, "Model-1.2_build")).st_mode & 0o777,
            0o700,
        )

        namespace_re = self.module._NAMESPACE_RE
        self.assertIsNotNone(namespace_re.fullmatch("Model-1.2_build"))
        self.assertIsNotNone(namespace_re.fullmatch("a" * 8))
        self.assertIsNotNone(namespace_re.fullmatch("a" * 128))
        for bad in ("short", "-leading", "has/slash", "a" * 129, "", "space pad"):
            self.assertIsNone(
                namespace_re.fullmatch(bad), f"namespace accepted: {bad!r}"
            )


class FileValidationTests(unittest.TestCase):
    def test_real_loader_rejects_bad_data(self):
        module, _ = _load_manager_module()
        payload = b"kv-block" * 8
        full = payload + hashlib.sha256(payload).digest()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chunk.bin"
            path.write_bytes(full)
            self.assertEqual(bytes(module._read_payload(str(path), len(payload))), payload)
            for bad in (full[:-3], full+b"x", b"x"+full[1:]):
                path.write_bytes(bad)
                with self.assertRaises(OSError):
                    module._read_payload(str(path), len(payload))


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.module, _ = _load_manager_module()
        self.worker = self.module.NvmeDirectWorker.__new__(self.module.NvmeDirectWorker)
        self.worker._namespace_dir = "/unused"
        self.worker._capacity_lock = self.module.threading.Lock()
        self.worker._capacity_bytes = 1000
        self.worker._required_free_bytes = 100
        self.worker._used_bytes = 0

    def test_quota_reservation_survives_until_restart(self):
        with patch.object(self.module.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10000)):
            self.worker._check_capacity(750)
            # The payload has not been committed yet; its reservation still counts.
            with self.assertRaisesRegex(OSError, "capacity reached"):
                self.worker._check_capacity(251)
            self.assertEqual(self.worker._used_bytes, 750)
            self.worker._check_capacity(250)
            self.assertEqual(self.worker._used_bytes, 1000)

    def test_disk_reserve_rejection_does_not_charge_quota(self):
        with patch.object(self.module.shutil, "disk_usage", return_value=types.SimpleNamespace(free=800)):
            with self.assertRaisesRegex(OSError, "free-space reserve reached"):
                self.worker._check_capacity(701)
            self.assertEqual(self.worker._used_bytes, 0)
            self.worker._check_capacity(700)
            self.assertEqual(self.worker._used_bytes, 700)


class ModuleSurfaceTests(unittest.TestCase):
    def test_module_exposes_pure_helpers_and_classes(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertTrue(set(_PURE_NAMES).issubset(funcs))
        classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        for name in (
            "NvmeFileLoadStoreSpec",
            "NvmeDirectManager",
            "NvmeDirectWorker",
            "NvmeDirectOffloadingSpec",
        ):
            self.assertIn(name, classes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
