# SPDX-License-Identifier: AGPL-3.0-only
from probe import *
for deg in [False,True]:
 c=Case(17,32,degenerate=deg);c.baseline();c.candidate();torch.cuda.synchronize()
 print(json.dumps({'degenerate':deg,'output':metrics(c.o1,c.o0),'state':metrics(c.s1,c.s0),'per_token_max':(c.o1.float()-c.o0.float()).abs().flatten(1).max(1).values.tolist()}),flush=True)
 if not deg: timed(c)
