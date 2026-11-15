import re, sys, statistics as st
from datetime import datetime
def ts(l): return datetime.strptime(l[:26], "%Y-%m-%dT%H:%M:%S.%f").timestamp()
L=[l.rstrip("\n") for l in open(sys.argv[1]) if re.match(r"\d{4}-", l)]
ev=[]
for l in L:
    t=ts(l)
    if "DFLASH WIDTH" in l:
        m=re.search(r"n_active=(\d+)", l); ev.append((t,"W",int(m.group(1))))
    elif "Chunked prefill start" in l:
        m=re.search(r"start: (\d+) prompt", l); ev.append((t,"S",int(m.group(1))))
    elif "PC_WRITE_FLOOR" in l or "resumes at recorded progress" in l: ev.append((t,"R",0))
    elif re.search(r"Prefill chunk \d+/\d+ tokens", l):
        m=re.search(r"Prefill chunk (\d+)/(\d+)", l); ev.append((t,"C",int(m.group(1))))
    elif "Prefill first token" in l: ev.append((t,"F",0))
# pure decode step intervals outside any prefill
inpf=False; last=None; pure={}
for t,k,v in ev:
    if k=="S" and v>4000: inpf=True; last=None
    elif k=="F": inpf=False; last=None
    elif k=="W":
        if not inpf and last and last[1]==v: pure.setdefault(v,[]).append(t-last[0])
        last=(t,v)
for n,x in sorted(pure.items()):
    x=[a for a in x if a<0.5]
    print(f"pure decode step n_active={n}: n={len(x)} median {st.median(x)*1000:.1f} ms mean {st.mean(x)*1000:.1f}")
# per big prefill: chunk durations and number of decode steps inside each chunk
cur=None
for i,(t,k,v) in enumerate(ev):
    if k=="S" and v>4000: cur=dict(n=v,t0=t,tc=t,w=[],r=[]); 
    elif cur is not None:
        if k=="W": cur["w"].append((t,v))
        elif k=="R": cur["r"].append(t)
        elif k=="C" and v>0:
            ws=cur["w"]; dur=t-cur["tc"]
            gaps=[b[0]-a[0] for a,b in zip(ws,ws[1:])]
            # decomposition where resume lines exist
            dec=[]; sl=[]
            rs=cur["r"]
            for (tw,_) in ws:
                nxt=[r for r in rs if r>tw]
                prv=[r for r in rs if r<tw]
                if nxt and nxt[0]-tw<0.3: dec.append(nxt[0]-tw)
                if prv and tw-prv[-1]<0.6: sl.append(tw-prv[-1])
            s=f"prompt {cur['n']} chunk->{v}: {dur:.3f}s steps {len(ws)} n_act {ws[0][1] if ws else '-'}"
            if gaps: s+=f" step-to-step median {st.median(gaps)*1000:.0f} ms"
            if dec: s+=f" | width->resume median {st.median(dec)*1000:.1f} ms (n={len(dec)}) resume->width median {st.median(sl)*1000:.1f}"
            print(s)
            cur["tc"]=t; cur["w"]=[]; cur["r"]=[]
        elif k=="F": cur=None
