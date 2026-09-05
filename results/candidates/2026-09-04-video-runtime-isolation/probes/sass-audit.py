# SPDX-License-Identifier: Apache-2.0
import collections,hashlib,json,re,subprocess,sys
result={}
for path,arch in zip(sys.argv[1::2],sys.argv[2::2]):
    proc=subprocess.Popen(['cuobjdump','--dump-sass','--gpu-architecture',arch,path],stdout=subprocess.PIPE,text=True)
    current=None;lines=[]; functions=collections.defaultdict(list)
    def retain():
        if current and lines:
            functions[current].append({'instructions':len(lines),'sha256':hashlib.sha256('\n'.join(lines).encode()).hexdigest()})
    for line in proc.stdout:
        if 'Function :' in line:
            retain();current=line.split('Function :',1)[1].strip();lines=[]
        elif re.match(r'\s*/\*[0-9a-fA-F]+\*/',line):
            # Normalize layout and instruction address, retaining opcodes and
            # encoded control words. No GPU instructions are executed.
            lines.append(' '.join(re.sub(r'^\s*/\*[0-9a-fA-F]+\*/\s*','',line).split()))
        elif re.match(r'\s*/\* 0x',line):
            lines.append(' '.join(line.split()))
    retain()
    if proc.wait()!=0: raise RuntimeError('cuobjdump failed')
    result[path]={'architecture':arch,'functions':dict(functions)}
print(json.dumps(result,sort_keys=True))
