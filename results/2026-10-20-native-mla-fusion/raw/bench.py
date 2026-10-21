#!/usr/bin/env python3
"""Reproducible appliance benchmark for OpenAI-compatible LLM servers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROTOCOL_VERSION = "1.1.0"
HERE = Path(__file__).resolve().parent
SOURCE_FILES = (
    "audit_code.py",
    "bench.py",
    "compare.py",
    "configure.py",
    "prompts.json",
    "receipt.py",
    "report.py",
    "rigmark",
)


def normalise_base_url(value: str) -> str:
    value = value.rstrip("/")
    if value.endswith("/v1"):
        value = value[:-3]
    return value


def validate_base_url(value: str) -> str:
    value = normalise_base_url(value)
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("base URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("put credentials in the API-key environment variable, not the URL")
    return value


def safe_label(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if not rendered:
        raise ValueError("label must contain a letter or number")
    return rendered


def nonce(comparison_id: str, *parts: object) -> str:
    source = ":".join((PROTOCOL_VERSION, comparison_id, *(str(part) for part in parts)))
    return hashlib.sha256(source.encode()).hexdigest()[:32]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def chunk_timed_decode_rate(
    completion_tokens: int,
    first_output: float,
    last_output: float,
    measured_events: int,
) -> tuple[float, float]:
    if completion_tokens > 1 and measured_events < 2:
        raise RuntimeError(
            "server buffered the completion into one measurable SSE event; "
            "decode rate is unavailable"
        )
    window = last_output - first_output
    if completion_tokens > 1 and window <= 0:
        raise RuntimeError("decode timing window is not measurable")
    window = max(window, 0.0)
    rate = 0.0 if completion_tokens <= 1 else (completion_tokens - 1) / window
    return window, rate


class Client:
    def __init__(self, base_url: str, api_key: str, timeout: float):
        self.base_url = validate_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=self._headers(),
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def stream(
        self,
        path: str,
        payload: dict[str, Any],
        record_events: bool = False,
        on_first_output: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Open one streaming request and time it.

        ``record_events`` additionally keeps the arrival time of every
        measurable SSE event (relative to the request start) together with the
        absolute monotonic start/first-output/finish instants, which the
        staggered-arrival suite needs to relate streams to each other.
        ``on_first_output`` is called once, when the first measurable output
        arrives; it must not raise.
        """
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers=self._headers(),
        )
        started = time.monotonic()
        first = None
        first_visible = None
        last = None
        usage: dict[str, Any] = {}
        finish_reason = None
        measured_chunks: list[str] = []
        output_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        event_seconds: list[float] = []
        stream_done_marker = False

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                encoded = line[5:].strip()
                if encoded == "[DONE]":
                    stream_done_marker = True
                    break
                event = json.loads(encoded)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                if (
                    len(choices) != 1
                    or not isinstance(choices[0], dict)
                    or choices[0].get("index", 0) != 0
                ):
                    raise RuntimeError(
                        "stream returned multiple or non-zero-index choices"
                    )
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                reasoning = "".join(
                    value
                    for key in ("reasoning", "reasoning_content")
                    if isinstance((value := delta.get(key)), str) and value
                )
                content = delta.get("content")
                if not isinstance(content, str):
                    content = ""
                if not content and isinstance(choice.get("text"), str):
                    content = choice["text"]
                measured = reasoning + content
                if measured:
                    now = time.monotonic()
                    if first is None:
                        first = now
                        if on_first_output is not None:
                            on_first_output()
                    if content:
                        first_visible = first_visible or now
                    last = now
                    if record_events:
                        event_seconds.append(round(now - started, 6))
                    measured_chunks.append(measured)
                    reasoning_chunks.append(reasoning)
                    output_chunks.append(content)

        finished = time.monotonic()
        if first is None or last is None:
            raise RuntimeError("stream emitted no measurable output")
        if "completion_tokens" not in usage or "prompt_tokens" not in usage:
            raise RuntimeError("server did not return final token usage")

        for field in ("completion_tokens", "prompt_tokens"):
            if (
                not isinstance(usage[field], int)
                or isinstance(usage[field], bool)
                or usage[field] < 0
            ):
                raise RuntimeError(
                    f"server returned a non-integer {field}: {usage[field]!r}"
                )
        completion = usage["completion_tokens"]
        prompt = usage["prompt_tokens"]
        measured_events = len(measured_chunks)
        decode_window, decode_rate = chunk_timed_decode_rate(
            completion, first, last, measured_events
        )
        rendered = "".join(measured_chunks)
        output = "".join(output_chunks)
        reasoning = "".join(reasoning_chunks)
        timeline: dict[str, Any] = {}
        if record_events:
            timeline = {
                "started_monotonic_seconds": round(started, 6),
                "first_output_monotonic_seconds": round(first, 6),
                "finished_monotonic_seconds": round(finished, 6),
                "event_seconds": event_seconds,
            }
        return {
            **timeline,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "ttft_seconds": round(first - started, 6),
            "time_to_first_visible_seconds": (
                None if first_visible is None else round(first_visible - started, 6)
            ),
            "time_to_last_output_seconds": round(last - started, 6),
            "decode_seconds": round(decode_window, 6),
            "decode_tokens_per_second": round(decode_rate, 3),
            "measured_sse_events": measured_events,
            "wall_seconds": round(finished - started, 6),
            "response_tail_seconds": round(max(finished - last, 0), 6),
            "stream_done_marker": stream_done_marker,
            "finish_reason": finish_reason,
            "output": output,
            "output_characters": len(output),
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "reasoning_characters": len(reasoning),
            "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
            "stream_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        }


def load_prompts(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        path = HERE / name
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def archival_identity(path: Path = HERE / ".git_archival.txt") -> dict[str, Any] | None:
    """Return the commit embedded by git archive when .git is unavailable."""
    try:
        values = dict(
            line.split(":", 1)
            for line in path.read_text().splitlines()
            if ":" in line
        )
    except OSError:
        return None
    revision = values.get("node", "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        return None
    return {
        "repository_revision": revision.lower(),
        "repository_dirty": None,
        "repository_source": "git-archive",
        "repository_source_sha256": source_fingerprint(),
    }


def git_identity() -> dict[str, Any]:
    try:
        root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=HERE,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        ).resolve()
        if root != HERE:
            raise RuntimeError("benchmark is inside an unrelated Git worktree")
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=HERE,
            stderr=subprocess.DEVNULL,
        )
        identity: dict[str, Any] = {
            "repository_revision": revision,
            "repository_dirty": bool(status),
            "repository_source": "git-worktree",
            "repository_source_sha256": source_fingerprint(),
        }
        if status:
            diff = subprocess.check_output(
                ["git", "diff", "--binary", "HEAD"],
                cwd=HERE,
                stderr=subprocess.DEVNULL,
            )
            untracked = subprocess.check_output(
                ["git", "ls-files", "--others", "--exclude-standard", "-z"],
                cwd=HERE,
                stderr=subprocess.DEVNULL,
            )
            worktree = hashlib.sha256(status + b"\0" + diff)
            for raw_path in sorted(path for path in untracked.split(b"\0") if path):
                path = HERE / os.fsdecode(raw_path)
                worktree.update(b"\0untracked\0" + raw_path + b"\0")
                if path.is_symlink():
                    worktree.update(os.fsencode(os.readlink(path)))
                elif path.is_file():
                    worktree.update(path.read_bytes())
            identity["repository_worktree_sha256"] = worktree.hexdigest()
        return identity
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        return archival_identity() or {
            "repository_revision": "unknown",
            "repository_dirty": None,
            "repository_source": "unknown",
        }


def summarise(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {
        "median": round(statistics.median(values), 3),
        "minimum": round(min(values), 3),
        "maximum": round(max(values), 3),
        "p90": round(percentile(values, 0.9), 3),
    }


def build_chat_payload(
    model: str,
    system: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    protected = {
        "model", "messages", "temperature", "top_p", "seed", "max_tokens",
        "stream", "stream_options", "n", "best_of", "stop",
        "max_completion_tokens", "min_tokens", "ignore_eos",
        "response_format", "guided_json", "guided_regex", "guided_choice",
        "use_beam_search",
    }
    conflicts = sorted(protected.intersection(extra_body))
    if conflicts:
        raise ValueError(
            "--extra-body cannot override fixed fields: " + ", ".join(conflicts)
        )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    payload.update(extra_body)
    return payload


def run_decode(
    client: Client,
    model: str,
    prompts: dict[str, Any],
    runs: int,
    max_tokens: int,
    seed: int,
    extra_body: dict[str, Any],
    comparison_id: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, prompt in prompts["workloads"].items():
        rows = []
        print(f"decode/{name}: {runs} runs", flush=True)
        for index in range(runs):
            request_nonce = nonce(comparison_id, "decode", name, index + 1)
            payload = build_chat_payload(
                model,
                prompts["system"],
                f"Request nonce: {request_nonce}\n\n{prompt}",
                max_tokens,
                seed,
                extra_body,
            )
            row = client.stream("/v1/chat/completions", payload)
            row["run"] = index + 1
            if name == "structured":
                row["completion_validation"] = validate_structured_row(row)
            else:
                row["completion_validation"] = validate_visible_output(row)
            rows.append(row)
            print(
                f"  {index + 1}: {row['decode_tokens_per_second']:.1f} tok/s, "
                f"TTFT {row['ttft_seconds']:.3f}s, "
                f"{row['completion_tokens']} tokens",
                flush=True,
            )
        output[name] = {
            "runs": rows,
            "decode_tokens_per_second": summarise(rows, "decode_tokens_per_second"),
            "ttft_seconds": summarise(rows, "ttft_seconds"),
            "time_to_last_output_seconds": summarise(
                rows, "time_to_last_output_seconds"
            ),
            "wall_seconds": summarise(rows, "wall_seconds"),
            "completion_gate": {
                "passed": sum(
                    1 for row in rows if row["completion_validation"]["valid"]
                ),
                "total": len(rows),
            },
        }
    return output


def validate_structured_output(output: str) -> dict[str, Any]:
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        return {"valid": False, "error": f"invalid JSON: {error.msg}"}
    if not isinstance(value, list) or len(value) != 50:
        return {"valid": False, "error": "expected an array of 50 objects"}
    for index, item in enumerate(value, start=1):
        if (
            not isinstance(item, dict)
            or set(item) != {"index", "square"}
            or type(item["index"]) is not int
            or type(item["square"]) is not int
            or item["index"] != index
            or item["square"] != index * index
        ):
            return {"valid": False, "error": f"incorrect object at index {index}"}
    return {"valid": True}


def normal_stream_end(row: dict[str, Any]) -> dict[str, Any] | None:
    reason = row.get("finish_reason")
    if reason != "stop":
        return {"valid": False, "error": f"non-normal finish reason: {reason!r}"}
    if "stream_done_marker" in row and row["stream_done_marker"] is not True:
        return {"valid": False, "error": "stream ended without a [DONE] marker"}
    return None


def validate_structured_row(row: dict[str, Any]) -> dict[str, Any]:
    failure = normal_stream_end(row)
    return failure or validate_structured_output(row.get("output", ""))


def validate_visible_output(row: dict[str, Any]) -> dict[str, Any]:
    if not row.get("output", "").strip():
        return {"valid": False, "error": "no visible answer"}
    failure = normal_stream_end(row)
    if failure:
        return failure
    return {"valid": True}


def exact_token_ids(
    client: Client,
    model: str,
    target: int,
    unit: str,
    request_nonce: str,
) -> list[int]:
    prefix_body = client.json(
        "/tokenize",
        {
            "model": model,
            "prompt": f"Unique cold-prefill nonce {request_nonce}.\n",
            "add_special_tokens": False,
        },
    )
    unit_body = client.json(
        "/tokenize",
        {"model": model, "prompt": unit, "add_special_tokens": False},
    )
    prefix_tokens = prefix_body.get("tokens") or []
    unit_tokens = unit_body.get("tokens") or []
    if not all(type(token) is int for token in [*prefix_tokens, *unit_tokens]):
        raise RuntimeError("tokenizer returned a non-integer token ID")
    if not prefix_tokens or not unit_tokens:
        raise RuntimeError("tokenizer returned an empty prefix or prefill unit")
    if len(prefix_tokens) > target:
        raise RuntimeError(
            f"prefill depth {target} is too small for the cache-busting prefix"
        )
    tokens = prefix_tokens[:target]
    while len(tokens) < target:
        tokens.extend(unit_tokens[:target - len(tokens)])
    return tokens


def prefill_once(client: Client, model: str, tokens: list[int]) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": tokens,
        "add_special_tokens": False,
        "max_tokens": 8,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    row = client.stream("/v1/completions", payload)
    requested = len(tokens)
    reported = row["prompt_tokens"]
    if reported != requested:
        raise RuntimeError(
            "prefill token-count mismatch: requested "
            f"{requested}, server reported {reported}; refusing to label this sample"
        )
    row["requested_prompt_tokens"] = requested
    row["effective_prefill_tokens_per_second"] = round(
        requested / max(row["ttft_seconds"], 1e-9), 3
    )
    return row


def run_prefill(
    client: Client,
    model: str,
    prompts: dict[str, Any],
    depths: list[int],
    runs: int,
    comparison_id: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for depth in depths:
        print(f"prefill/{depth}: {runs} cold/immediate-replay pairs", flush=True)
        cold_rows = []
        warm_rows = []
        for index in range(runs):
            tokens = exact_token_ids(
                client,
                model,
                depth,
                prompts["prefill_unit"],
                nonce(comparison_id, "prefill", depth, index + 1),
            )
            cold = prefill_once(client, model, tokens)
            warm = prefill_once(client, model, tokens)
            cold["run"] = index + 1
            warm["run"] = index + 1
            cold_rows.append(cold)
            warm_rows.append(warm)
            print(
                f"  {index + 1}: cold "
                f"{cold['effective_prefill_tokens_per_second']:.1f} tok/s "
                f"({cold['ttft_seconds']:.3f}s), warm "
                f"{warm['effective_prefill_tokens_per_second']:.1f} tok/s "
                f"({warm['ttft_seconds']:.3f}s)",
                flush=True,
            )
        output[str(depth)] = {
            "cold": {
                "runs": cold_rows,
                "effective_prefill_tokens_per_second": summarise(
                    cold_rows, "effective_prefill_tokens_per_second"
                ),
                "ttft_seconds": summarise(cold_rows, "ttft_seconds"),
            },
            "warm_replay": {
                "runs": warm_rows,
                "effective_prefill_tokens_per_second": summarise(
                    warm_rows, "effective_prefill_tokens_per_second"
                ),
                "ttft_seconds": summarise(warm_rows, "ttft_seconds"),
            },
        }
    return output


def run_concurrency(
    client: Client,
    model: str,
    prompts: dict[str, Any],
    levels: list[int],
    rounds: int,
    max_tokens: int,
    seed: int,
    extra_body: dict[str, Any],
    workload: str,
    comparison_id: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    prompt = prompts["workloads"][workload]
    for level in levels:
        print(f"concurrency/{level}: {rounds} rounds of {workload}", flush=True)
        round_rows = []
        all_streams = []
        for round_index in range(rounds):
            barrier = threading.Barrier(level)

            def invoke(stream_index: int) -> dict[str, Any]:
                request_nonce = nonce(
                    comparison_id,
                    "concurrency",
                    workload,
                    level,
                    round_index + 1,
                    stream_index + 1,
                )
                payload = build_chat_payload(
                    model,
                    prompts["system"],
                    f"Request nonce: {request_nonce}\n\n{prompt}",
                    max_tokens,
                    seed + stream_index,
                    extra_body,
                )
                barrier.wait()
                return client.stream("/v1/chat/completions", payload)

            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=level) as executor:
                streams = list(executor.map(invoke, range(level)))
            wall = time.monotonic() - started
            completion = sum(row["completion_tokens"] for row in streams)
            aggregate = completion / max(wall, 1e-9)
            round_row = {
                "round": round_index + 1,
                "wall_seconds": round(wall, 6),
                "completion_tokens": completion,
                "aggregate_end_to_end_tokens_per_second": round(aggregate, 3),
                "streams": streams,
            }
            round_rows.append(round_row)
            all_streams.extend(streams)
            print(
                f"  {round_index + 1}: aggregate {aggregate:.1f} tok/s, "
                "median stream "
                f"{statistics.median(row['decode_tokens_per_second'] for row in streams):.1f} "
                "tok/s",
                flush=True,
            )
        output[str(level)] = {
            "rounds": round_rows,
            "aggregate_end_to_end_tokens_per_second": summarise(
                round_rows, "aggregate_end_to_end_tokens_per_second"
            ),
            "per_stream_decode_tokens_per_second": summarise(
                all_streams, "decode_tokens_per_second"
            ),
            "per_stream_ttft_seconds": summarise(all_streams, "ttft_seconds"),
        }
    return output


def completion_payload(
    model: str, tokens: list[int], max_tokens: int, seed: int
) -> dict[str, Any]:
    """Exact-token completion request used for the long staggered request."""
    return {
        "model": model,
        "prompt": tokens,
        "add_special_tokens": False,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }


def delivery_gaps(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Absolute event instants and the intervals between consecutive events."""
    started = row["started_monotonic_seconds"]
    instants = [started + offset for offset in row["event_seconds"]]
    return instants, [b - a for a, b in zip(instants, instants[1:])]


def stall_analysis(
    row: dict[str, Any], window_start: float, window_end: float
) -> dict[str, Any]:
    """Delivery stalls of one recorded stream around another request.

    The whole-stream p95/max intervals describe how smoothly the stream was
    delivered overall.  The arrival-window figures only consider intervals that
    end inside ``[window_start, window_end]`` (the other request's start and
    first output), which is where a scheduler pauses incumbents to admit a
    prefill.  ``spanning_gap_seconds`` is the single interval during which the
    other request started, or ``None`` when this stream had already finished.
    """
    instants, gaps = delivery_gaps(row)
    spanning = None
    window = []
    for start, end, gap in zip(instants, instants[1:], gaps):
        if start <= window_start < end:
            spanning = gap
        if window_start <= end <= window_end:
            window.append(gap)
    return {
        "events": len(instants),
        "median_gap_seconds": round(statistics.median(gaps), 6) if gaps else 0.0,
        "p95_gap_seconds": round(percentile(gaps, 0.95), 6) if gaps else 0.0,
        "max_gap_seconds": round(max(gaps), 6) if gaps else 0.0,
        "spanning_gap_seconds": None if spanning is None else round(spanning, 6),
        "arrival_window_gaps": len(window),
        "arrival_window_p95_gap_seconds": (
            round(percentile(window, 0.95), 6) if window else None
        ),
        "arrival_window_max_gap_seconds": (
            round(max(window), 6) if window else None
        ),
    }


def wait_for_events(
    events: list[threading.Event],
    futures: list[Any],
    timeout: float,
) -> None:
    """Block until every event is set; fail fast if a producer already failed."""
    deadline = time.monotonic() + timeout
    for event in events:
        while not event.wait(0.25):
            for future in futures:
                if future.done() and future.exception() is not None:
                    raise RuntimeError(
                        "an incumbent stream failed before its first output: "
                        f"{future.exception()}"
                    )
            if time.monotonic() > deadline:
                raise TimeoutError("incumbent streams produced no output in time")


def staggered_decode_first_round(
    client: Client,
    model: str,
    prompts: dict[str, Any],
    level: int,
    round_index: int,
    settings: dict[str, Any],
    seed: int,
    extra_body: dict[str, Any],
    comparison_id: str,
) -> dict[str, Any]:
    """C-1 short incumbents decode; one long-context request arrives."""
    incumbents = level - 1
    workload = settings["staggered_workload"]
    prompt = prompts["workloads"][workload]
    depth = settings["staggered_depth"]
    delay = settings["staggered_delay_seconds"]
    solo_tokens = exact_token_ids(
        client, model, depth, prompts["prefill_unit"],
        nonce(comparison_id, "staggered", "decode-first", "solo", level, round_index),
    )
    mixed_tokens = exact_token_ids(
        client, model, depth, prompts["prefill_unit"],
        nonce(comparison_id, "staggered", "decode-first", "arrival", level, round_index),
    )
    solo = client.stream(
        "/v1/completions",
        completion_payload(model, solo_tokens, settings["staggered_arrival_tokens"], seed),
        record_events=True,
    )
    if solo["prompt_tokens"] != depth:
        raise RuntimeError("staggered solo prompt token count does not match depth")

    first_events = [threading.Event() for _ in range(incumbents)]
    barrier = threading.Barrier(incumbents)
    arrival_box: dict[str, Any] = {}

    def incumbent(index: int) -> dict[str, Any]:
        request_nonce = nonce(
            comparison_id, "staggered", "decode-first", "incumbent",
            level, round_index, index + 1,
        )
        payload = build_chat_payload(
            model, prompts["system"], f"Request nonce: {request_nonce}\n\n{prompt}",
            settings["staggered_incumbent_tokens"], seed + index, extra_body,
        )
        barrier.wait()
        row = client.stream(
            "/v1/chat/completions", payload,
            record_events=True, on_first_output=first_events[index].set,
        )
        row["stream"] = index + 1
        return row

    with ThreadPoolExecutor(max_workers=incumbents) as executor:
        futures = [executor.submit(incumbent, index) for index in range(incumbents)]
        wait_for_events(first_events, futures, client.timeout)
        time.sleep(delay)
        arrival_box["started"] = time.monotonic()
        newcomer = client.stream(
            "/v1/completions",
            completion_payload(model, mixed_tokens, settings["staggered_arrival_tokens"], seed),
            record_events=True,
        )
        incumbent_rows = [future.result() for future in futures]

    if newcomer["prompt_tokens"] != depth:
        raise RuntimeError("staggered newcomer prompt token count does not match depth")
    arrival = newcomer["started_monotonic_seconds"]
    newcomer_first = newcomer["first_output_monotonic_seconds"]
    last_first_output = max(row["first_output_monotonic_seconds"] for row in incumbent_rows)
    overlap_valid = all(row["finished_monotonic_seconds"] > arrival for row in incumbent_rows)
    stalls = [stall_analysis(row, arrival, newcomer_first) for row in incumbent_rows]
    for row, stall in zip(incumbent_rows, stalls):
        row["stall"] = stall
    window_gaps = [
        stall["arrival_window_max_gap_seconds"]
        for stall in stalls
        if stall["arrival_window_max_gap_seconds"] is not None
    ]
    return {
        "round": round_index,
        "overlap_valid": overlap_valid,
        "arrival_after_last_incumbent_first_output_seconds": round(arrival - last_first_output, 6),
        "incumbents_finished_before_newcomer_first_output": sum(
            row["finished_monotonic_seconds"] <= newcomer_first for row in incumbent_rows
        ),
        "newcomer_ttft_seconds": newcomer["ttft_seconds"],
        "newcomer_solo_ttft_seconds": solo["ttft_seconds"],
        "newcomer_ttft_ratio_vs_solo": round(
            newcomer["ttft_seconds"] / max(solo["ttft_seconds"], 1e-9), 3
        ),
        "newcomer_decode_tokens_per_second": newcomer["decode_tokens_per_second"],
        "newcomer_wall_seconds": newcomer["wall_seconds"],
        "incumbent_max_p95_gap_seconds": max(stall["p95_gap_seconds"] for stall in stalls),
        "incumbent_max_arrival_window_gap_seconds": (
            max(window_gaps) if window_gaps else None
        ),
        "incumbent_median_decode_tokens_per_second": round(
            statistics.median(row["decode_tokens_per_second"] for row in incumbent_rows), 3
        ),
        "solo": solo,
        "newcomer": newcomer,
        "incumbents": incumbent_rows,
    }


def staggered_prefill_first_round(
    client: Client,
    model: str,
    prompts: dict[str, Any],
    level: int,
    round_index: int,
    settings: dict[str, Any],
    seed: int,
    extra_body: dict[str, Any],
    comparison_id: str,
    long_solo_ttft: float,
) -> dict[str, Any]:
    """One long-context request is prefilling; C-1 short requests arrive."""
    newcomers = level - 1
    workload = settings["staggered_workload"]
    prompt = prompts["workloads"][workload]
    depth = settings["staggered_depth"]
    delay = settings["staggered_delay_seconds"]
    long_tokens = exact_token_ids(
        client, model, depth, prompts["prefill_unit"],
        nonce(comparison_id, "staggered", "prefill-first", "incumbent", level, round_index),
    )
    solo_nonce = nonce(comparison_id, "staggered", "prefill-first", "solo", level, round_index)
    solo = client.stream(
        "/v1/chat/completions",
        build_chat_payload(
            model, prompts["system"], f"Request nonce: {solo_nonce}\n\n{prompt}",
            settings["staggered_arrival_tokens"], seed, extra_body,
        ),
        record_events=True,
    )

    first_event = threading.Event()

    def incumbent() -> dict[str, Any]:
        return client.stream(
            "/v1/completions",
            completion_payload(model, long_tokens, settings["staggered_incumbent_tokens"], seed),
            record_events=True, on_first_output=first_event.set,
        )

    def newcomer(index: int) -> dict[str, Any]:
        request_nonce = nonce(
            comparison_id, "staggered", "prefill-first", "arrival",
            level, round_index, index + 1,
        )
        payload = build_chat_payload(
            model, prompts["system"], f"Request nonce: {request_nonce}\n\n{prompt}",
            settings["staggered_arrival_tokens"], seed + index, extra_body,
        )
        barrier.wait()
        row = client.stream("/v1/chat/completions", payload, record_events=True)
        row["stream"] = index + 1
        return row

    barrier = threading.Barrier(newcomers)
    with ThreadPoolExecutor(max_workers=level) as executor:
        long_future = executor.submit(incumbent)
        time.sleep(delay)
        arrived_during_prefill = not first_event.is_set() and not long_future.done()
        newcomer_rows: list[dict[str, Any]] = []
        if arrived_during_prefill:
            futures = [executor.submit(newcomer, index) for index in range(newcomers)]
            newcomer_rows = [future.result() for future in futures]
        long_row = long_future.result()

    if long_row["prompt_tokens"] != depth:
        raise RuntimeError("staggered long prompt token count does not match depth")
    long_first = long_row["first_output_monotonic_seconds"]
    overlap_valid = bool(newcomer_rows) and all(
        row["started_monotonic_seconds"] < long_first for row in newcomer_rows
    )
    result: dict[str, Any] = {
        "round": round_index,
        "overlap_valid": overlap_valid,
        "newcomers_started_during_prefill": sum(
            row["started_monotonic_seconds"] < long_first for row in newcomer_rows
        ),
        "long_ttft_seconds": long_row["ttft_seconds"],
        "long_solo_ttft_seconds": round(long_solo_ttft, 6),
        "long_ttft_ratio_vs_solo": round(
            long_row["ttft_seconds"] / max(long_solo_ttft, 1e-9), 3
        ),
        "short_solo_ttft_seconds": solo["ttft_seconds"],
        "solo": solo,
        "incumbent": long_row,
        "newcomers": newcomer_rows,
    }
    if newcomer_rows:
        ttfts = [row["ttft_seconds"] for row in newcomer_rows]
        median_ttft = statistics.median(ttfts)
        result.update({
            "newcomer_median_ttft_seconds": round(median_ttft, 6),
            "newcomer_max_ttft_seconds": round(max(ttfts), 6),
            "newcomer_ttft_ratio_vs_solo": round(
                median_ttft / max(solo["ttft_seconds"], 1e-9), 3
            ),
            "newcomer_median_decode_tokens_per_second": round(
                statistics.median(row["decode_tokens_per_second"] for row in newcomer_rows), 3
            ),
        })
    return result


DECODE_FIRST_KEYS = (
    "newcomer_ttft_seconds",
    "newcomer_ttft_ratio_vs_solo",
    "newcomer_decode_tokens_per_second",
    "incumbent_max_arrival_window_gap_seconds",
    "incumbent_max_p95_gap_seconds",
    "incumbent_median_decode_tokens_per_second",
)
PREFILL_FIRST_KEYS = (
    "newcomer_median_ttft_seconds",
    "newcomer_max_ttft_seconds",
    "newcomer_ttft_ratio_vs_solo",
    "long_ttft_seconds",
    "long_ttft_ratio_vs_solo",
)


def summarise_valid_rounds(
    rounds: list[dict[str, Any]], keys: tuple[str, ...]
) -> dict[str, Any]:
    valid = [row for row in rounds if row["overlap_valid"]]
    summary: dict[str, Any] = {
        "rounds": rounds,
        "valid_rounds": len(valid),
        "total_rounds": len(rounds),
    }
    for key in keys:
        present = [row for row in valid if row.get(key) is not None]
        summary[key] = summarise(present, key) if present else None
    return summary


def run_staggered(
    client: Client,
    model: str,
    prompts: dict[str, Any],
    settings: dict[str, Any],
    seed: int,
    extra_body: dict[str, Any],
    comparison_id: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    rounds = settings["staggered_runs"]
    for level in settings["staggered"]:
        print(
            f"staggered/{level}: {rounds} rounds, {level - 1} incumbents + 1 arrival, "
            f"{settings['staggered_depth']:,}-token long request",
            flush=True,
        )
        decode_first = []
        prefill_first = []
        for round_index in range(1, rounds + 1):
            row = staggered_decode_first_round(
                client, model, prompts, level, round_index, settings, seed,
                extra_body, comparison_id,
            )
            decode_first.append(row)
            print(
                f"  {round_index}: decode-first newcomer TTFT {row['newcomer_ttft_seconds']:.3f}s "
                f"({row['newcomer_ttft_ratio_vs_solo']:.2f}x solo), incumbent stall "
                + (
                    f"{row['incumbent_max_arrival_window_gap_seconds']:.3f}s"
                    if row["incumbent_max_arrival_window_gap_seconds"] is not None
                    else "n/a"
                )
                + f" (whole-stream p95 gap {row['incumbent_max_p95_gap_seconds']:.3f}s)"
                + ("" if row["overlap_valid"] else " [no overlap]"),
                flush=True,
            )
            row = staggered_prefill_first_round(
                client, model, prompts, level, round_index, settings, seed,
                extra_body, comparison_id, decode_first[-1]["newcomer_solo_ttft_seconds"],
            )
            prefill_first.append(row)
            if row["overlap_valid"]:
                print(
                    f"  {round_index}: prefill-first short TTFT median "
                    f"{row['newcomer_median_ttft_seconds']:.3f}s "
                    f"({row['newcomer_ttft_ratio_vs_solo']:.2f}x solo), long TTFT "
                    f"{row['long_ttft_seconds']:.3f}s ({row['long_ttft_ratio_vs_solo']:.2f}x solo)",
                    flush=True,
                )
            else:
                print(
                    f"  {round_index}: prefill-first arrivals did not overlap the prefill "
                    "(increase --staggered-depth or lower --staggered-delay)",
                    flush=True,
                )
        output[str(level)] = {
            "decode_first": summarise_valid_rounds(decode_first, DECODE_FIRST_KEYS),
            "prefill_first": summarise_valid_rounds(prefill_first, PREFILL_FIRST_KEYS),
        }
    return output


def comma_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def validate_prefill_depths(depths: list[int], context_limit: int) -> None:
    for depth in depths:
        if depth + 8 > context_limit:
            raise ValueError(
                f"prefill depth {depth} plus 8 generated tokens exceeds "
                f"the declared context limit {context_limit}"
            )


def write_result(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"result: {output}", flush=True)


def load_metadata(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not value:
        raise ValueError("metadata must be a non-empty JSON object")
    required = {
        "hardware", "topology", "model", "model_revision", "quantisation",
        "kv_cache_dtype", "serving_engine", "context_limit",
        "competing_traffic",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError("metadata is missing required fields: " + ", ".join(missing))

    def contains_placeholder(item: Any) -> bool:
        if isinstance(item, str):
            return "CHANGE ME" in item.upper()
        if isinstance(item, dict):
            return any(contains_placeholder(child) for child in item.values())
        if isinstance(item, list):
            return any(contains_placeholder(child) for child in item)
        return False

    if contains_placeholder(value):
        raise ValueError(
            "metadata still contains CHANGE ME placeholders; run configure.py"
        )
    context_limit = value["context_limit"]
    if isinstance(context_limit, bool) or not isinstance(context_limit, int):
        raise ValueError("metadata context_limit must be a whole number")
    if context_limit < 1:
        raise ValueError("metadata context_limit must be greater than zero")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=4096)
    parser.add_argument(
        "--prefill-depths",
        type=comma_ints,
        default=comma_ints("8192,32768,65536"),
    )
    parser.add_argument("--prefill-runs", type=int, default=3)
    parser.add_argument("--concurrency", type=comma_ints, default=comma_ints("1,2,4"))
    parser.add_argument("--concurrency-runs", type=int, default=3)
    parser.add_argument("--concurrency-tokens", type=int, default=256)
    parser.add_argument("--concurrency-workload", choices=("code", "prose"), default="code")
    parser.add_argument(
        "--staggered", type=comma_ints, default=[],
        help="total concurrency levels (incumbents + 1 arrival, each >= 2) for the "
        "staggered-arrival suite; omitted by default",
    )
    parser.add_argument("--staggered-runs", type=int, default=3)
    parser.add_argument(
        "--staggered-depth", type=int, default=32768,
        help="exact prompt tokens of the long request (cache-busting, like prefill)",
    )
    parser.add_argument(
        "--staggered-incumbent-tokens", type=int, default=1024,
        help="output cap of the streams that are already running when the arrival happens",
    )
    parser.add_argument(
        "--staggered-arrival-tokens", type=int, default=256,
        help="output cap of the arriving request(s)",
    )
    parser.add_argument(
        "--staggered-delay", type=float, default=1.0,
        help="seconds between every incumbent's first output (decode-first) or the long "
        "request's start (prefill-first) and the arrival",
    )
    parser.add_argument("--staggered-workload", choices=("code", "prose"), default="code")
    parser.add_argument("--extra-body", default="{}", help="JSON merged into every chat request")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-concurrency", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    if args.runs < 1 or args.prefill_runs < 1 or args.concurrency_runs < 1:
        parser.error("run counts must be positive")
    if args.staggered:
        if args.staggered_runs < 1:
            parser.error("--staggered-runs must be positive")
        if any(level < 2 for level in args.staggered) or len(set(args.staggered)) != len(args.staggered):
            parser.error("--staggered levels must be distinct and at least 2")
        if args.staggered_incumbent_tokens < 2 or args.staggered_arrival_tokens < 2:
            parser.error("staggered token caps must be at least 2")
        if not 0 <= args.staggered_delay <= 60:
            parser.error("--staggered-delay must be between 0 and 60 seconds")
    try:
        base_url = validate_base_url(args.base_url)
        filename_label = safe_label(args.label)
        metadata = load_metadata(args.metadata)
        extra_body = json.loads(args.extra_body)
        if not args.skip_prefill:
            validate_prefill_depths(args.prefill_depths, metadata["context_limit"])
        if args.staggered:
            validate_prefill_depths([args.staggered_depth], metadata["context_limit"])
    except (json.JSONDecodeError, OSError, ValueError) as error:
        parser.error(str(error))
    if not isinstance(extra_body, dict):
        parser.error("--extra-body must be a JSON object")

    api_key = os.environ.get(args.api_key_env, "")
    client = Client(base_url, api_key, args.timeout)
    prompts, prompts_sha256 = load_prompts(HERE / "prompts.json")
    model = args.model
    if model == "auto":
        models = client.json("/v1/models").get("data") or []
        if len(models) != 1:
            parser.error("--model auto requires the endpoint to advertise exactly one model")
        model = str(models[0]["id"])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("results") / f"{filename_label}-{timestamp}.json"
    repository = git_identity()
    result: dict[str, Any] = {
        "schema": 1,
        "protocol": {
            "version": PROTOCOL_VERSION,
            **repository,
            "prompts_version": prompts["version"],
            "prompts_sha256": prompts_sha256,
        },
        "run": {
            "label": args.label,
            "comparison_id": args.comparison_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "appliance": metadata,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "settings": {
            "runs": args.runs,
            "decode_tokens": args.decode_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": args.seed,
            "extra_body": extra_body,
            "prefill_depths": [] if args.skip_prefill else args.prefill_depths,
            "prefill_runs": args.prefill_runs,
            "concurrency": [] if args.skip_concurrency else args.concurrency,
            "concurrency_runs": args.concurrency_runs,
            "concurrency_tokens": args.concurrency_tokens,
            "concurrency_workload": args.concurrency_workload,
            "staggered": args.staggered,
            "staggered_runs": args.staggered_runs,
            "staggered_depth": args.staggered_depth,
            "staggered_incumbent_tokens": args.staggered_incumbent_tokens,
            "staggered_arrival_tokens": args.staggered_arrival_tokens,
            "staggered_delay_seconds": args.staggered_delay,
            "staggered_workload": args.staggered_workload,
        },
    }

    try:
        result["decode"] = run_decode(
            client,
            model,
            prompts,
            args.runs,
            args.decode_tokens,
            args.seed,
            extra_body,
            args.comparison_id,
        )
        if not args.skip_prefill:
            result["prefill"] = run_prefill(
                client,
                model,
                prompts,
                args.prefill_depths,
                args.prefill_runs,
                args.comparison_id,
            )
        if not args.skip_concurrency:
            result["concurrency"] = run_concurrency(
                client,
                model,
                prompts,
                args.concurrency,
                args.concurrency_runs,
                args.concurrency_tokens,
                args.seed,
                extra_body,
                args.concurrency_workload,
                args.comparison_id,
            )
        if args.staggered:
            result["staggered"] = run_staggered(
                client,
                model,
                prompts,
                result["settings"],
                args.seed,
                extra_body,
                args.comparison_id,
            )
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as error:
        result["error"] = str(error)
        write_result(result, output)
        raise

    result["run"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_result(result, output)
    try:
        from report import print_report, save_report

        card = save_report(result, output)
        print(f"card: {card}")
        print()
        print_report(result, output)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"warning: could not render terminal report: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
