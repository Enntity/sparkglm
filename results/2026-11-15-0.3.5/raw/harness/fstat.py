import json,sys,statistics as st,glob,os
root=sys.argv[1]
for arm in sys.argv[2:]:
    for k in ("cold-16k","cold-32k","field-16k","field-32k"):
        fs=sorted(glob.glob(f"{root}/{arm}/{k}/field-guide-r*.json"))
        if not fs: continue
        rows=[]
        for f in fs:
            d=json.load(open(f)); r=d["requests"]
            gaps=[]; 
            for q in r:
                ts=[e["t"] for e in q["events"]]
                gaps.append(max([b-a for a,b in zip(ts,ts[1:])] or [0]))
            rows.append(dict(wall=d["wall_s"],first=[q["first_token_s"] for q in r],end=[q["ended_s"] for q in r],gap=gaps,agg=d.get("aggregate_decode_tok_s")))
        med=lambda xs: st.median(xs)
        n=len(rows[0]["first"])
        print(f"{arm:6} {k:9} walls {[round(x['wall'],2) for x in rows]} median {med([x['wall'] for x in rows]):.2f}")
        for key in ("first","end","gap"):
            print(f"       {key:5}", [round(med([x[key][i] for x in rows]),2) for i in range(n)])
