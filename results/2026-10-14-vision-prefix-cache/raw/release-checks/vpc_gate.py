#!/usr/bin/env python3
"""vpc_gate.py: hardware gate for vision prefix caching (Enntity/atlas feat/vision-prefix-cache).

  vpc_gate.py TAG conv           ~90K-token conversation with an image in turn 1, then four
                                 short turns: per-turn TTFT, cached tokens, greedy text
  vpc_gate.py TAG cold CONV N    turn N of conv-CONV.json sent alone (run on a fresh server):
                                 the full-recompute answer to compare token for token
  vpc_gate.py TAG swap           two-image order swap, a same-position image swap, and a re-send
  vpc_gate.py TAG swapcold SWAP  the swapped requests of swap-SWAP.json alone (fresh server)
  vpc_gate.py TAG race           image request B arrives while image request A (~60K tokens, image
                                 at the end, so its rows are spliced in A's last chunk) is still
                                 prefilling: before the admission hold, A described B's image

Greedy, thinking off. Every image is 448x448, so all of them expand to the same
placeholder run: before content keys, a later image reused an earlier one's KV.
Writes <phase>-TAG.json in the working directory. Needs code.txt (make_code.py) beside it.
"""
import base64, io, json, os, sys, time, urllib.request

from PIL import Image, ImageDraw

URL = "http://127.0.0.1:8893/v1/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))


def png(shape, color):
    im = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(im)
    box = (104, 104, 344, 344)
    if shape == "circle":
        d.ellipse(box, fill=color)
    elif shape == "square":
        d.rectangle(box, fill=color)
    else:
        d.polygon([(224, 96), (352, 352), (96, 352)], fill=color)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


P, Q, R = png("circle", "red"), png("square", "blue"), png("triangle", "green")


def text(t):
    return {"type": "text", "text": t}


def stream(messages, max_tokens=80):
    body = {"model": "glm-5.3-flash-atlas", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; out = ""; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                out += piece
    return {"ttft": round(ttft or -1, 2), "wall": round(time.time() - t0, 2), "text": out.strip(),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


def log(name, r):
    print(f"{name}: ttft {r['ttft']} s, prompt {r['prompt_tokens']}, cached {r['cached']}: {r['text'][:110]!r}", flush=True)


FOLLOW_UPS = [
    "What color was the shape in the screenshot? One word.",
    "Describe the screenshot in two sentences, then name one Rust file from the codebase.",
    "Name a fruit of that color. One word.",
    "Describe the screenshot again, in two different sentences.",
]


TEXT_FOLLOW_UPS = [
    "Name one Rust file from the codebase. Just the path.",
    "What does that file do? Two sentences.",
    "Name one function in it. Just the name.",
    "Describe the codebase again, in two different sentences.",
]


def conv(tag, image=True):
    code = open(os.path.join(HERE, "code.txt")).read()
    first = ([text("Here is our codebase:\n\n" + code), P,
              text("Also, here is a screenshot from the app. What shape and color does it show? A few words.")]
             if image else "Here is our codebase:\n\n" + code + "\n\nWhat language is it written in? A few words.")
    follow = FOLLOW_UPS if image else TEXT_FOLLOW_UPS
    msgs = [{"role": "system", "content": "You are a helpful coding assistant. Answer briefly."},
            {"role": "user", "content": first}]
    turns = []
    for i in range(len(follow) + 1):
        r = stream(msgs)
        r["messages"] = len(msgs)
        turns.append(r); log(f"turn {i + 1}", r)
        msgs.append({"role": "assistant", "content": r["text"]})
        if i < len(follow):
            msgs.append({"role": "user", "content": follow[i]})
    json.dump({"tag": tag, "turns": turns, "messages": msgs}, open(f"conv-{tag}.json", "w"), indent=1)


def cold(tag, conv_tag, n):
    saved = json.load(open(f"conv-{conv_tag}.json"))
    msgs = saved["messages"][:saved["turns"][n - 1]["messages"]]
    r = stream(msgs); log(f"cold turn {n}", r)
    warm = saved["turns"][n - 1]
    r["equal_to_warm"] = r["text"] == warm["text"]
    print(f"turn {n} warm == cold: {r['equal_to_warm']}", flush=True)
    json.dump({"tag": tag, "conv": conv_tag, "turn": n, "cold": r, "warm": warm}, open(f"cold-{tag}.json", "w"), indent=1)


TWO = "There are two images. Name the shape and color of the first image, then of the second. Format: first: <color> <shape>; second: <color> <shape>."
ONE = "What shape and color is in this image? Answer as: <color> <shape>."


def swap_requests():
    lead = open(os.path.join(HERE, "code.txt")).read()[:8000]
    sys_ = {"role": "system", "content": "Answer exactly in the requested format."}
    single = lambda img: [sys_, {"role": "user", "content": [text("Context:\n" + lead), img, text(ONE)]}]
    return {
        "order_PQ": [sys_, {"role": "user", "content": [P, Q, text(TWO)]}],
        "order_QP": [sys_, {"role": "user", "content": [Q, P, text(TWO)]}],
        "single_P": single(P),
        "single_R": single(R),
        "single_P_again": single(P),
    }


def swap(tag):
    out = {}
    for name, msgs in swap_requests().items():
        r = stream(msgs, max_tokens=40); log(name, r); out[name] = r
    json.dump({"tag": tag, "results": out}, open(f"swap-{tag}.json", "w"), indent=1)


def swapcold(tag, swap_tag):
    warm = json.load(open(f"swap-{swap_tag}.json"))["results"]
    reqs = swap_requests(); out = {}
    for name in ("order_QP", "single_R"):
        r = stream(reqs[name], max_tokens=40); log(f"cold {name}", r)
        r["equal_to_warm"] = r["text"] == warm[name]["text"]
        print(f"{name} warm == cold: {r['equal_to_warm']}", flush=True)
        out[name] = r
    json.dump({"tag": tag, "swap": swap_tag, "results": out}, open(f"swapcold-{tag}.json", "w"), indent=1)


def race(tag, first=None, second=None):
    """A = first (image at the end of ~47K tokens), B = second, sent 3 s later."""
    import threading
    (pa, na), (pb, nb) = first or (P, ("red", "circle")), second or (Q, ("blue", "square"))
    code = open(os.path.join(HERE, "code.txt")).read()[:190000]
    # The template renders a message's images before its text, so the long
    # context goes in the system message to put A's image in its last chunk.
    a = [{"role": "system", "content": "Answer exactly in the requested format.\n\nContext:\n" + code},
         {"role": "user", "content": [pa, text(ONE)]}]
    b = [{"role": "system", "content": "Answer exactly in the requested format."},
         {"role": "user", "content": [pb, text(ONE)]}]
    out = {}
    def go(name, msgs):
        out[name] = stream(msgs, max_tokens=40); out[name]["done_at"] = time.time()
    ka, kb = "A_" + "_".join(na), "B_" + "_".join(nb)
    ta = threading.Thread(target=go, args=(ka, a)); ta.start()
    time.sleep(3)
    tb = threading.Thread(target=go, args=(kb, b)); tb.start()
    ta.join(); tb.join()
    for name, r in out.items():
        log(name, r)
    ok_a = all(w in out[ka]["text"].lower() for w in na)
    ok_b = all(w in out[kb]["text"].lower() for w in nb)
    print(f"race: A {'correct' if ok_a else 'WRONG'}, B {'correct' if ok_b else 'WRONG'}", flush=True)
    json.dump({"tag": tag, "results": out, "a_correct": ok_a, "b_correct": ok_b}, open(f"race-{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    tag, phase = sys.argv[1], sys.argv[2]
    if phase == "conv":
        conv(tag)
    elif phase == "textconv":  # control: the same shape without the image
        conv(tag, image=False)
    elif phase == "cold":
        cold(tag, sys.argv[3], int(sys.argv[4]))
    elif phase == "swap":
        swap(tag)
    elif phase == "swapcold":
        swapcold(tag, sys.argv[3])
    elif phase == "race":
        race(tag)
    elif phase == "race2":  # A's right answer, B's contamination and the default guess all differ
        race(tag, (Q, ("blue", "square")), (R, ("green", "triangle")))
    else:
        sys.exit(__doc__)
