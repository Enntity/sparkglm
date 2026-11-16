#!/usr/bin/env python3
"""vision_eq.py run TAG | compare TAG_A TAG_B

Greedy vision outputs and per-token logprobs against the running server, alone
and concurrently with long text decodes, so an engine change that touches the
vision path can be checked byte for byte. Fixtures: the suite's digits.png and
colors.mp4, plus a large still and a long, large video made here (deterministic),
so encodes span tens of thousands of patches.
Writes ./vision-TAG.json. URL/MODEL from VEQ_URL / VEQ_MODEL.
"""
import base64, concurrent.futures as cf, hashlib, json, os, pathlib, subprocess, sys, urllib.request

URL = os.environ.get("VEQ_URL", "http://127.0.0.1:8893/v1/chat/completions")
MODEL = os.environ.get("VEQ_MODEL", "glm-5.3-flash-atlas")
KEY = os.environ.get("VEQ_KEY", "sk-local-llm")
HERE = pathlib.Path(__file__).resolve().parent
FIX = pathlib.Path(os.environ.get("VEQ_FIXTURES", HERE / "veq-fixtures"))


def fixtures():
    FIX.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw
    big = FIX / "big.png"
    if not big.exists():
        img = Image.new("RGB", (1792, 1344), "white")
        d = ImageDraw.Draw(img)
        for i in range(24):
            d.rectangle([i * 70, 40 + (i % 5) * 60, i * 70 + 50, 300 + (i % 7) * 90], fill=((i * 37) % 256, (i * 91) % 256, (i * 53) % 256))
        for row in range(12):
            d.text((60, 520 + row * 60), f"row {row}: inventory item {row * 7 + 3} costs {row * 13 + 9} dollars", fill="black")
        img.save(big)
    vid = FIX / "long.mp4"
    if not vid.exists():
        frames = FIX / "frames"
        frames.mkdir(exist_ok=True)
        for f in range(48):
            img = Image.new("RGB", (1280, 960), ((f * 5) % 256, 40, 255 - (f * 5) % 256))
            d = ImageDraw.Draw(img)
            d.rectangle([40 + f * 20, 300, 240 + f * 20, 500], fill="yellow")
            d.text((50, 50), f"frame {f}", fill="white")
            img.save(frames / f"f{f:03d}.png")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "4", "-i", str(frames / "f%03d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid)], check=True)
    return {
        "digits": ("image/png", pathlib.Path(os.path.expanduser("~/qual-20261103/kit/qualify/fixtures/digits.png"))),
        "colors": ("video/mp4", pathlib.Path(os.path.expanduser("~/qual-20261103/kit/qualify/fixtures/colors.mp4"))),
        "big": ("image/png", big),
        "long": ("video/mp4", vid),
    }


def data_url(mime, path):
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def ask(content, max_tokens):
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": "low",
            "logprobs": True, "top_logprobs": 0, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    c = r["choices"][0]
    lp = [t.get("logprob") for t in ((c.get("logprobs") or {}).get("content") or [])]
    msg = c["message"]
    return {"text": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "finish": c.get("finish_reason"), "logprobs": lp, "usage": r.get("usage")}


def cases(fx):
    vis = lambda kind, name, prompt: [{"type": f"{kind}_url", f"{kind}_url": {"url": data_url(*fx[name])}}, {"type": "text", "text": prompt}]
    return {
        "digits": (vis("image", "digits", "Describe this image in detail, then state the number it shows."), 256),
        "big": (vis("image", "big", "Transcribe every line of text in this image, in order."), 512),
        "colors": (vis("video", "colors", "Describe what happens in this video, frame by frame."), 256),
        "long": (vis("video", "long", "Describe how the scene changes across this video, and list every frame number you can read."), 512),
    }


TEXT = [("essay", "Write a long, detailed essay on the history of the printing press, with many dates and names.", 900),
        ("code", "Write a complete Python implementation of a B-tree with insert, delete and search, with comments.", 900)]


def run(tag):
    fx = fixtures()
    cs = cases(fx)
    out = {"alone": {}, "mixed": {}}
    for name, (content, mt) in cs.items():
        out["alone"][name] = ask(content, mt)
        print(name, "alone", out["alone"][name]["finish"], len(out["alone"][name]["logprobs"]), flush=True)
    jobs = [(n, c, m) for n, (c, m) in cs.items()] + [(n, p, m) for n, p, m in TEXT]
    with cf.ThreadPoolExecutor(len(jobs)) as pool:
        futs = {pool.submit(ask, c, m): n for n, c, m in jobs}
        for f in cf.as_completed(futs):
            out["mixed"][futs[f]] = f.result()
            print(futs[f], "mixed", out["mixed"][futs[f]]["finish"], flush=True)
    pathlib.Path(f"vision-{tag}.json").write_text(json.dumps(out, indent=1))
    print("wrote", f"vision-{tag}.json")


def compare(a, b):
    A, B = (json.load(open(f"vision-{t}.json")) for t in (a, b))
    same = total = 0
    for mode in ("alone", "mixed"):
        for name in A[mode]:
            x, y = A[mode][name], B[mode].get(name)
            total += 1
            ok = y is not None and x["text"] == y["text"] and x["reasoning"] == y["reasoning"] and x["logprobs"] == y["logprobs"]
            same += ok
            if not ok and y is not None:
                d = next((i for i, (p, q) in enumerate(zip(x["logprobs"], y["logprobs"])) if p != q), None)
                print(f"DIFF {mode}/{name}: text_same={x['text'] == y['text']} reasoning_same={x['reasoning'] == y['reasoning']} first_logprob_diff={d}")
    print(f"{a} vs {b}: {same}/{total} identical (text, reasoning and every token logprob)")


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "compare": lambda: compare(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
