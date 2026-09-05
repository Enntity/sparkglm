# SPDX-License-Identifier: Apache-2.0
import hashlib,json,struct,sys

def inspect(path):
    with open(path,'rb') as f:
        head=f.read(64)
        if head[:6] != b'\x7fELF\x02\x01': raise ValueError('Expected ELF64 LE')
        shoff=struct.unpack_from('<Q',head,40)[0]
        entsize,count,namesindex=struct.unpack_from('<HHH',head,58)
        f.seek(shoff)
        headers=[struct.unpack('<IIQQQQIIQQ',f.read(entsize)) for _ in range(count)]
        nameshead=headers[namesindex]; f.seek(nameshead[4]); names=f.read(nameshead[5])
        sections={}
        for h in headers:
            name=names[h[0]:].split(b'\0',1)[0].decode()
            if not name or h[1]==8: continue
            f.seek(h[4]); remaining=h[5]; digest=hashlib.sha256()
            while remaining:
                part=f.read(min(4*1024*1024,remaining)); digest.update(part); remaining-=len(part)
            sections[name]={'bytes':h[5],'sha256':digest.hexdigest()}
        return sections
print(json.dumps({p:inspect(p) for p in sys.argv[1:]},sort_keys=True))
