# SPDX-License-Identifier: AGPL-3.0-only
from probe import *
shim=C.CDLL(str(ROOT/'libatlas_mango_flash.so'))
f=shim.atlas_mango_flash_prefill
f.argtypes=[P]*10+[C.c_uint64]*3+[I]*2+[C.c_float]*2+[P]
assert shim.atlas_mango_flash_abi_version()==1
for t in [2048,4096,4100]:
 c=Case(t,32);c.candidate();reference=c.o1.clone();state=c.s1.clone()
 packed=torch.empty(3*c.gate.numel(),device='cuda',dtype=torch.bfloat16)
 ws=torch.empty_like(c.workspace)
 auxsize=((32*128*128*4+32*t*2+127)//128)*128+20
 aux=torch.empty(auxsize,device='cuda',dtype=torch.uint8)
 saved=[x.clone() for x in [c.qkv,c.gate,c.rawbeta,c.a,c.bias]]
 c.s1.copy_(c.initial)
 args=ptrs(c.qkv,c.gate,c.rawbeta,c.a,c.bias,c.s1,c.o1,packed,ws,aux)
 sizes=[packed.numel()*2,ws.numel(),aux.numel()]
 for idx in range(3):
  wrong=sizes.copy();wrong[idx]=1
  assert f(*args,*wrong,t,32,1/math.sqrt(128),-5.,stream)!=0
  assert torch.equal(c.s1,c.initial), 'invalid capacity mutated state'
 assert f(*args,*sizes,t,32,float('inf'),-5.,stream)!=0
 check(f(*args,*sizes,t,32,1/math.sqrt(128),-5.,stream));torch.cuda.synchronize()
 assert torch.equal(c.o1,reference) and torch.equal(c.s1,state), 'shim differs from qualified probe'
 for x,y in zip([c.qkv,c.gate,c.rawbeta,c.a,c.bias],saved): assert torch.equal(x,y), 'input modified'
 print(json.dumps({'tokens':t,'exact_shim_parity':True,'input_immutable':True,'capacity_rejected':True}),flush=True)
