#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Capture four real OpenAI streams and render their timing as an MP4 grid."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import textwrap
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

TOPICS = (
    ("SYSTEMS", "distributed systems engineering"),
    ("SPACE", "surprising facts about the solar system"),
    ("OCEAN", "deep-ocean science and exploration"),
    ("ROBOTICS", "practical robotics and embodied AI"),
)

COUNTING_STREAMS = (
    (
        "STREAM 01",
        "Output integers starting at 1 and increasing by 1, separated only "
        "by spaces. Continue without stopping.",
    ),
    (
        "STREAM 02",
        "Output integers starting at 1001 and increasing by 1, separated only "
        "by spaces. Continue without stopping.",
    ),
    (
        "STREAM 03",
        "Output integers starting at 2001 and increasing by 1, separated only "
        "by spaces. Continue without stopping.",
    ),
    (
        "STREAM 04",
        "Output integers starting at 3001 and increasing by 1, separated only "
        "by spaces. Continue without stopping.",
    ),
)


def _stream_one(
    index: int,
    topic: tuple[str, str],
    barrier: threading.Barrier,
    zero: float,
    args: argparse.Namespace,
    results: list[dict[str, Any] | None],
) -> None:
    label, subject = topic
    if args.prompt_style == "counting":
        label, prompt = COUNTING_STREAMS[index]
        subject = "structured counting decode"
    else:
        prompt = (
            "Write a dense numbered field guide containing concise facts about "
            f"{subject}. Start immediately with item 1. Use one short item per line, "
            "no introduction, "
            "no conclusion, and continue until the response limit."
        )
    if args.prompt_tokens:
        # The numbered/punctuated context line tokenizes to roughly 12 GLM
        # tokens. The API usage record below remains the authoritative count.
        facts = "\n".join(
            f"context-{index}-{fact}: alpha beta gamma delta"
            for fact in range(max(1, args.prompt_tokens // 12))
        )
        salt = f"Benchmark salt: {args.prompt_salt}\n" if args.prompt_salt else ""
        prompt = salt + (
            "Read the unique reference context below before answering the final "
            "request.\n"
            f"{facts}\n\nFinal request:\n{prompt}"
        )
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "top_p": 1,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if getattr(args, "cache_salt", None):
        # Isolate KV reuse without changing the recorded prompt text/tokens.
        body["cache_salt"] = args.cache_salt
    if args.logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 0
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )

    events: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    error: str | None = None
    barrier.wait()
    target = zero + index * args.stagger_ms / 1000
    delay = target - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    scheduled = time.monotonic() - zero
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_s) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                item = json.loads(payload)
                if item.get("usage"):
                    usage = item["usage"]
                choices = item.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                text = (delta.get("reasoning_content") or "") + (
                    delta.get("content") or ""
                )
                logprob_content = (choice.get("logprobs") or {}).get("content") or []
                if text or logprob_content:
                    events.append(
                        {
                            "t": round(time.monotonic() - zero, 6),
                            "text": text,
                            "tokens": len(logprob_content),
                        }
                    )
    except Exception as exc:  # capture the failure in the replay artifact
        error = f"{type(exc).__name__}: {exc}"

    ended = time.monotonic() - zero
    completion_tokens = int(usage.get("completion_tokens") or 0)
    counted_tokens = sum(int(event["tokens"]) for event in events)
    token_count_method = "stream_logprobs"
    if completion_tokens and events and counted_tokens == 0:
        # Logprobs materially slow this speculative-decode path. Preserve the
        # exact SSE timestamps and exact final usage, and apportion the live
        # counter over real deltas according to their emitted character share.
        token_count_method = "final_usage_weighted_by_delta_chars"
        weights = [max(len(event["text"].encode("utf-8")), 1) for event in events]
        total_weight = sum(weights)
        assigned = 0
        cumulative_weight = 0
        for event, weight in zip(events, weights):
            cumulative_weight += weight
            target = round(completion_tokens * cumulative_weight / total_weight)
            event["tokens"] = target - assigned
            assigned = target
    elif completion_tokens and events and counted_tokens != completion_tokens:
        events[-1]["tokens"] += completion_tokens - counted_tokens
    first = events[0]["t"] if events else None
    decode_s = max(ended - first, 1e-9) if first is not None else None
    results[index] = {
        "index": index,
        "label": label,
        "subject": subject,
        "scheduled_s": round(scheduled, 6),
        "first_token_s": first,
        "ttft_s": round(first - scheduled, 6) if first is not None else None,
        "ended_s": round(ended, 6),
        "completion_tokens": completion_tokens,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "decode_tok_s": (
            round(max(completion_tokens - 1, 0) / decode_s, 6)
            if decode_s is not None
            else None
        ),
        "token_count_method": token_count_method,
        "events": events,
        "error": error,
    }


def capture(args: argparse.Namespace) -> None:
    if args.streams != 4:
        raise SystemExit("this visualizer currently requires --streams 4")
    zero = time.monotonic()
    barrier = threading.Barrier(args.streams)
    results: list[dict[str, Any] | None] = [None] * args.streams
    threads = [
        threading.Thread(
            target=_stream_one,
            args=(index, TOPICS[index], barrier, zero, args, results),
            daemon=True,
        )
        for index in range(args.streams)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lanes = [lane for lane in results if lane is not None]
    first = min(
        (lane["first_token_s"] for lane in lanes if lane["first_token_s"] is not None),
        default=None,
    )
    ended = max((lane["ended_s"] for lane in lanes), default=0.0)
    total = sum(int(lane["completion_tokens"]) for lane in lanes)
    aggregate = (
        sum(max(int(lane["completion_tokens"]) - 1, 0) for lane in lanes)
        / max(ended - first, 1e-9)
        if first is not None
        else 0.0
    )
    artifact = {
        "schema": 1,
        "model": args.model,
        "streams": args.streams,
        "max_tokens": args.max_tokens,
        "prompt_style": args.prompt_style,
        "prompt_salt": args.prompt_salt,
        "cache_salt": getattr(args, "cache_salt", None),
        "prompt_tokens_target": args.prompt_tokens,
        "stagger_ms": args.stagger_ms,
        "recipe_label": args.recipe_label,
        "engine_commit": args.engine_commit,
        "wall_s": round(ended, 6),
        "total_completion_tokens": total,
        "aggregate_decode_tok_s": round(aggregate, 6),
        "requests": lanes,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(destination),
                "wall_s": artifact["wall_s"],
                "total_completion_tokens": total,
                "aggregate_decode_tok_s": artifact["aggregate_decode_tok_s"],
                "lanes": [
                    {
                        "label": lane["label"],
                        "ttft_s": lane["ttft_s"],
                        "decode_tok_s": lane["decode_tok_s"],
                        "completion_tokens": lane["completion_tokens"],
                        "error": lane["error"],
                    }
                    for lane in lanes
                ],
            }
        )
    )


def _font(path: str, size: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, size=size)


def _wrapped_lines(text: str, width: int) -> list[str]:
    clean = "".join(char if char >= " " or char == "\n" else " " for char in text)
    lines: list[str] = []
    for paragraph in clean.splitlines() or [""]:
        lines.extend(
            textwrap.wrap(
                paragraph,
                width=width,
                replace_whitespace=False,
                drop_whitespace=True,
            )
            or [""]
        )
    return lines


def _mix_color(start: str, end: str, amount: float) -> str:
    """Blend two RGB hex colors for a brief completion flash."""
    amount = min(max(amount, 0.0), 1.0)
    start_rgb = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(
        round(left + (right - left) * amount) for left, right in zip(start_rgb, end_rgb)
    )
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def render(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw

    artifact = json.loads(Path(args.input).read_text())
    lanes = artifact["requests"]
    actual_prompt_tokens = [int(lane.get("prompt_tokens") or 0) for lane in lanes]
    actual_prompt_tokens = [tokens for tokens in actual_prompt_tokens if tokens]
    prompt_tokens_per_stream = (
        round(sum(actual_prompt_tokens) / len(actual_prompt_tokens))
        if actual_prompt_tokens
        else 0
    )
    width, height = args.width, args.height
    fps = args.fps
    duration = max(float(lane["ended_s"]) for lane in lanes) + args.hold_s
    frame_count = math.ceil(duration * fps)

    mono_path = "/System/Library/Fonts/SFNSMono.ttf"
    sans_path = "/System/Library/Fonts/SFNS.ttf"
    title_font = _font(sans_path, 36)
    subtitle_font = _font(sans_path, 19)
    pane_title_font = _font(mono_path, 22)
    metric_font = _font(mono_path, 16)
    body_font = _font(mono_path, 17)
    footer_font = _font(mono_path, 19)
    done_font = _font(sans_path, 25)

    bg = "#07090d"
    panel = "#10151d"
    border = "#263140"
    text_color = "#dbe5f1"
    muted = "#7f91a8"
    done_green = "#38e8a3"
    done_panel = "#10291f"
    colors = ("#65d6ad", "#68a7ff", "#ffb454", "#d68cff")

    margin = 32
    top = 102
    bottom = 92
    gap = 18
    pane_w = (width - margin * 2 - gap) // 2
    pane_h = (height - top - bottom - gap) // 2
    positions = (
        (margin, top),
        (margin + pane_w + gap, top),
        (margin, top + pane_h + gap),
        (margin + pane_w + gap, top + pane_h + gap),
    )
    line_h = 23
    body_top_offset = 72
    visible_lines = max((pane_h - body_top_offset - 16) // line_h, 1)
    # SF Mono is roughly 10.5 px at this size. Keep extra room so punctuation
    # and wide glyphs cannot visually bleed into the neighboring pane.
    wrap_width = max((pane_w - 34) // 12, 20)

    command = [
        args.ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        args.output,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_index in range(frame_count):
            now = frame_index / fps
            image = Image.new("RGB", (width, height), bg)
            draw = ImageDraw.Draw(image)
            draw.text((margin, 24), "GLM-5.3 FLASH", font=title_font, fill="#f4f7fb")
            draw.text(
                (margin + 315, 34),
                (
                    "4 STAGGERED STREAMS  •  2× DGX SPARK"
                    if artifact.get("stagger_ms", 0)
                    else "4 LIVE STREAMS  •  2× DGX SPARK"
                ),
                font=subtitle_font,
                fill=muted,
            )

            total_live_tokens = 0
            active_firsts: list[float] = []
            for lane_index, lane in enumerate(lanes):
                x, y = positions[lane_index]
                color = colors[lane_index]
                ended = float(lane["ended_s"])
                done = now >= ended
                done_age = max(now - ended, 0.0)
                flash = max(1.0 - done_age / 0.9, 0.0) if done else 0.0
                pane_fill = (
                    _mix_color(done_panel, done_green, 0.18 * flash) if done else panel
                )
                pane_border = (
                    _mix_color(done_green, "#e2fff4", flash) if done else border
                )
                draw.rounded_rectangle(
                    (x, y, x + pane_w, y + pane_h),
                    radius=14,
                    fill=pane_fill,
                    outline=pane_border,
                    width=6 if done else 2,
                )
                if done:
                    # Completion should read as a state change from across the
                    # room, not as another small metric. Turn the entire header
                    # green and keep it latched for the rest of the replay.
                    draw.rounded_rectangle(
                        (x + 3, y + 3, x + pane_w - 3, y + 68),
                        radius=11,
                        fill=pane_border,
                    )
                    draw.rectangle(
                        (x + 3, y + 43, x + pane_w - 3, y + 68),
                        fill=pane_border,
                    )
                    draw.rectangle(
                        (x + 3, y + pane_h - 10, x + pane_w - 3, y + pane_h - 3),
                        fill=pane_border,
                    )
                events = [event for event in lane["events"] if float(event["t"]) <= now]
                stream_text = "".join(event["text"] for event in events)
                live_tokens = sum(int(event["tokens"]) for event in events)
                total_live_tokens += live_tokens
                first = lane["first_token_s"]
                if first is not None and now >= first:
                    active_firsts.append(float(first))
                if lane["error"]:
                    status = "ERROR"
                elif done:
                    status = "COMPLETE"
                elif events:
                    status = "STREAMING"
                elif now < float(lane["scheduled_s"]):
                    status = "NOT SENT"
                else:
                    spinner_index = int((now - float(lane["scheduled_s"])) * 8) % 4
                    spinner = "|/-\\"[spinner_index]
                    status = f"WAITING {spinner}"
                if first is not None and now > first:
                    rate = max(live_tokens - 1, 0) / max(now - float(first), 0.001)
                    if done:
                        rate = float(lane["decode_tok_s"] or 0.0)
                else:
                    rate = 0.0

                draw.text(
                    (x + 17, y + 13),
                    (
                        f"0{lane_index + 1}  {lane['label']}"
                        f"  •  +{float(lane['scheduled_s']):.1f}s"
                    ),
                    font=pane_title_font,
                    fill=bg if done else color,
                )
                metric = (
                    f"{status:<9} {live_tokens:>3}/"
                    f"{lane['completion_tokens']} tok  {rate:>5.1f} tok/s"
                )
                draw.text(
                    (x + 17, y + 43),
                    metric,
                    font=metric_font,
                    fill="#123229" if done else muted,
                )
                if done:
                    done_text = "DONE"
                    done_box = draw.textbbox((0, 0), done_text, font=done_font)
                    done_width = done_box[2] - done_box[0]
                    done_x = x + pane_w - done_width - 24
                    done_y = y + 16
                    draw.text(
                        (done_x, done_y),
                        done_text,
                        font=done_font,
                        fill=bg,
                    )
                lines = _wrapped_lines(stream_text, wrap_width)[-visible_lines:]
                for line_index, line in enumerate(lines):
                    draw.text(
                        (x + 17, y + body_top_offset + line_index * line_h),
                        line,
                        font=body_font,
                        fill=text_color,
                    )

            if active_firsts:
                aggregate_live = max(total_live_tokens - len(active_firsts), 0) / max(
                    now - min(active_firsts), 0.001
                )
            else:
                aggregate_live = 0.0
            if now >= max(float(lane["ended_s"]) for lane in lanes):
                aggregate_live = float(artifact["aggregate_decode_tok_s"])

            footer_y = height - 62
            identity = "  •  ".join(
                part
                for part in (
                    artifact.get("recipe_label", ""),
                    (
                        f"commit {artifact['engine_commit']}"
                        if artifact.get("engine_commit")
                        else ""
                    ),
                )
                if part
            )
            if identity:
                draw.text(
                    (margin, footer_y - 27),
                    identity,
                    font=metric_font,
                    fill="#65d6ad",
                )
            draw.text(
                (margin, footer_y),
                (
                    "EXL3 4bpw  •  TP2  •  DFlash2 k7  •  "
                    + (
                        f"~{prompt_tokens_per_stream / 1024:.1f}K actual "
                        "prompt/stream  •  "
                        if prompt_tokens_per_stream
                        else ""
                    )
                    + "measured SSE replay"
                ),
                font=footer_font,
                fill=muted,
            )
            right = (
                f"{now:05.1f}s   {total_live_tokens:>4} tok   "
                f"AGG {aggregate_live:>6.1f} tok/s"
            )
            right_box = draw.textbbox((0, 0), right, font=footer_font)
            draw.text(
                (width - margin - (right_box[2] - right_box[0]), footer_y),
                right,
                font=footer_font,
                fill="#f4f7fb",
            )
            process.stdin.write(image.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(
        json.dumps(
            {
                "output": args.output,
                "duration_s": round(duration, 3),
                "frames": frame_count,
                "aggregate_decode_tok_s": artifact["aggregate_decode_tok_s"],
            }
        )
    )


def _video_duration(path: str, ffprobe: str) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def stack(args: argparse.Namespace) -> None:
    """Stack two benchmark replays vertically and hold the earlier finish."""
    top_duration = _video_duration(args.top, args.ffprobe)
    bottom_duration = _video_duration(args.bottom, args.ffprobe)
    duration = max(top_duration, bottom_duration)
    top_pad = max(duration - top_duration, 0.0)
    bottom_pad = max(duration - bottom_duration, 0.0)
    panel = (
        f"scale={args.panel_width}:{args.panel_height}:"
        "force_original_aspect_ratio=decrease,"
        f"pad={args.panel_width}:{args.panel_height}:(ow-iw)/2:(oh-ih)/2:color=black"
    )
    filters = (
        f"[0:v]{panel},setpts=PTS-STARTPTS,"
        f"tpad=stop_mode=clone:stop_duration={top_pad:.6f}[top];"
        f"[1:v]{panel},setpts=PTS-STARTPTS,"
        f"tpad=stop_mode=clone:stop_duration={bottom_pad:.6f}[bottom];"
        "[top][bottom]vstack=inputs=2[out]"
    )
    subprocess.run(
        [
            args.ffmpeg,
            "-y",
            "-i",
            args.top,
            "-i",
            args.bottom,
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-an",
            "-r",
            str(args.fps),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            args.output,
        ],
        check=True,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "layout": "top-bottom",
                "duration_s": round(duration, 3),
                "top": args.top,
                "bottom": args.bottom,
            }
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--base-url", required=True)
    capture_parser.add_argument("--model", required=True)
    capture_parser.add_argument("--api-key", default="")
    capture_parser.add_argument("--streams", type=int, default=4)
    capture_parser.add_argument("--max-tokens", type=int, default=400)
    capture_parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="Approximate unique context tokens prepended to each stream.",
    )
    capture_parser.add_argument(
        "--prompt-salt",
        default="",
        help="Unique prefix salt that prevents cross-run prefix-cache hits.",
    )
    capture_parser.add_argument(
        "--stagger-ms",
        type=int,
        default=0,
        help="Delay each successive request by this many milliseconds.",
    )
    capture_parser.add_argument(
        "--prompt-style", choices=("counting", "field-guide"), default="counting"
    )
    capture_parser.add_argument("--recipe-label", default="")
    capture_parser.add_argument(
        "--cache-salt", default=None,
        help="KV-cache namespace; unlike prompt-salt this does not alter prompt tokens.",
    )
    capture_parser.add_argument("--engine-commit", default="")
    capture_parser.add_argument("--timeout-s", type=float, default=600)
    capture_parser.add_argument("--logprobs", action="store_true")
    capture_parser.add_argument("--output", required=True)
    capture_parser.set_defaults(func=capture)

    render_parser = commands.add_parser("render")
    render_parser.add_argument("--input", required=True)
    render_parser.add_argument("--output", required=True)
    render_parser.add_argument("--width", type=int, default=1920)
    render_parser.add_argument("--height", type=int, default=1080)
    render_parser.add_argument("--fps", type=int, default=20)
    render_parser.add_argument("--hold-s", type=float, default=2.0)
    render_parser.add_argument("--ffmpeg", default="ffmpeg")
    render_parser.set_defaults(func=render)

    stack_parser = commands.add_parser("stack")
    stack_parser.add_argument("--top", required=True)
    stack_parser.add_argument("--bottom", required=True)
    stack_parser.add_argument("--output", required=True)
    stack_parser.add_argument("--panel-width", type=int, default=1600)
    stack_parser.add_argument("--panel-height", type=int, default=900)
    stack_parser.add_argument("--fps", type=int, default=12)
    stack_parser.add_argument("--ffmpeg", default="ffmpeg")
    stack_parser.add_argument("--ffprobe", default="ffprobe")
    stack_parser.set_defaults(func=stack)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
