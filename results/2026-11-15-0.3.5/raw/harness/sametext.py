import json,sys,glob,os
# sametext.py ROOT A B: do arms A and B give the same text for every stream of every cold capture?
root,a,b=sys.argv[1:4]; same=diff=0
for fa in sorted(glob.glob(f"{root}/{a}/cold-*/*.json")):
    fb=fa.replace(f"/{a}/",f"/{b}/")
    if not os.path.exists(fb): continue
    ra=json.load(open(fa))["requests"]; rb=json.load(open(fb))["requests"]
    for x,y in zip(ra,rb):
        tx="".join(e["text"] for e in x["events"]); ty="".join(e["text"] for e in y["events"])
        if tx==ty and x["completion_tokens"]==y["completion_tokens"]: same+=1
        else:
            diff+=1; i=next((k for k,(p,q) in enumerate(zip(tx,ty)) if p!=q),min(len(tx),len(ty)))
            print("DIFF",fa.split(root)[1],x["label"],"at char",i,"of",len(tx),len(ty))
print(f"{a} vs {b}: {same} streams identical, {diff} differ")
