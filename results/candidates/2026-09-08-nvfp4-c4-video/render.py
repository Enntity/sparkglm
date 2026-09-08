# SPDX-License-Identifier: Apache-2.0
# Original measured-SSE comparison renderer; no synthetic timing.
import json,pathlib,math,subprocess,textwrap,sys
from PIL import Image,ImageDraw,ImageFont
root=pathlib.Path(sys.argv[1]);sets=json.loads((root/'comparison.json').read_text());fps=12;W,H=2560,1440
font='/System/Library/Fonts/SFNS.ttf';mono='/System/Library/Fonts/SFNSMono.ttf'
def f(n,m=False):return ImageFont.truetype(mono if m else font,n)
F={n:f(n) for n in [18,20,22,25,30,38]};M=f(17,True)
end=max(d['capture']['wall_s'] for d in sets)+4
out=root/'glm53-tp2-c4-progress-20260908.mp4'
proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
for frame in range(math.ceil(end*fps)):
 t=frame/fps;im=Image.new('RGB',(W,H),'#080e17');d=ImageDraw.Draw(im)
 d.text((30,20),'SparkGLM progress · four staggered streams',font=F[38],fill='#f3f6fb')
 d.text((30,70),'2 × DGX Spark · ~16K input per request · 400 output tokens · arrivals at 0 / 1 / 2 / 3 seconds',font=F[22],fill='#a8b9cb')
 d.text((2230,29),f'{t:05.1f}s  /  1×',font=F[25],fill='#a8b9cb')
 for c,item in enumerate(sets):
  a=item['capture'];x=30+(c%2)*1270;oy=(c//2)*640;color=['#92a8c9','#70b9f5','#5fe0ae'][c]
  d.text((x,115+oy),item['title'],font=F[30],fill=color)
  d.text((x,155+oy),item['subtitle'],font=F[20],fill='#bdcad8')
  d.text((x,183+oy),item['detail'],font=F[18],fill='#8296ac')
  delivered=0
  for j,l in enumerate(a['requests']):
   xx=x; x=xx+(j%2)*615; y=220+oy+(j//2)*215;done=t>=l['ended_s'];ev=[e for e in l['events'] if e['t']<=t];count=sum(e.get('tokens',0) for e in ev);count=l['completion_tokens'] if done else min(count,l['completion_tokens']);delivered+=count
   state='DONE' if done else ('STREAMING' if l['first_token_s'] is not None and t>=l['first_token_s'] else ('PREFILL / WAIT' if t>=l['scheduled_s'] else 'NOT SENT'))
   d.rounded_rectangle((x,y,x+606,y+195),radius=10,fill='#102b26' if done else '#111d2b',outline='#315647' if done else '#273b51')
   d.text((x+14,y+10),f'{j+1:02d}  {l["label"]}',font=F[22],fill=color);d.text((x+337,y+13),f'{state}  {count}/400',font=F[18],fill='#d5e5ee')
   txt=''.join(e['text'] for e in ev).replace('\n',' ');lines=textwrap.wrap(txt,width=50)[-4:]
   for k,line in enumerate(lines):d.text((x+14,y+45+k*23),line,font=M,fill='#cddbe8')
   d.rectangle((x+14,y+175,x+592,y+180),fill='#29394a');d.rectangle((x+14,y+175,x+14+578*count/400,y+180),fill=color)
   x=xx
  finished=t>=a['wall_s'];d.text((x,650+oy),f'{delivered:,} / 1,600 tokens delivered',font=F[25],fill=color)
  label=f'Finished in {a["wall_s"]:.2f}s' if finished else 'Processing measured stream events…'
  d.text((x+650,650+oy),label,font=F[25],fill='#f3f6fb')
 d.text((30,1360),'Historical baseline and prior capture vs current median of 3 runs. Configurations differ; this is end-to-end progress.',font=F[20],fill='#a8b9cb')
 d.text((30,1396),'Real SSE arrival times. Token animation uses server totals distributed across text deltas. Full elapsed time retained.',font=F[18],fill='#8296ac')
 if frame in [0,360,720,math.ceil(end*fps)-1]:im.save(root/f'preview-{frame}.png')
 proc.stdin.write(im.tobytes())
proc.stdin.close();assert proc.wait()==0
print(out)
