#!/usr/bin/env python3
"""vision_check.py: two image requests with known answers (pure-Python PNGs sent as data URLs)."""
import base64, json, struct, sys, time, urllib.request, zlib
URL = "http://127.0.0.1:8893/v1/chat/completions"
def png(w, h, px):
    raw = b"".join(b"\x00" + b"".join(bytes(px(x, y)) for x in range(w)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
def ask(img, q):
    body = {"model": "glm-5.3-flash-atlas", "max_tokens": 200, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()}},
                                                        {"type": "text", "text": q}]}]}
    t = time.time(); d = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
    return d["choices"][0]["message"].get("content") or "", round(time.time() - t, 2), d["usage"]["prompt_tokens"]
halves = png(256, 128, lambda x, y: (220, 30, 30) if x < 128 else (30, 60, 220))
grid = png(240, 240, lambda x, y: (0, 160, 0) if ((x // 60) + (y // 60)) % 2 == 0 else (250, 250, 250))
ok = 0
a, t, n = ask(halves, "This image has two halves. Name the color of the left half and the color of the right half, in that order, one word each.")
good = "red" in a.lower() and "blue" in a.lower() and a.lower().find("red") < a.lower().find("blue"); ok += good
print(f"halves: {'PASS' if good else 'FAIL'} {t}s prompt={n} answer={a.strip()[:120]!r}")
a, t, n = ask(grid, "This is a checkerboard. How many squares are in each row, and what two colors are used? Answer as: N, color1, color2.")
good = "4" in a and "green" in a.lower() and "white" in a.lower(); ok += good
print(f"grid: {'PASS' if good else 'FAIL'} {t}s prompt={n} answer={a.strip()[:120]!r}")
print(f"vision {ok}/2")
