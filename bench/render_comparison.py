#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Measured-SSE comparison renderer (SparkGLM's stacked C4 layout, N arms); no
# synthetic timing. Needs Pillow and ffmpeg.
# Usage: render_comparison.py COMPARISON.json OUT.mp4 "TITLE" [--preview-only]
# Fonts: RENDER_FONT / RENDER_MONO, else macOS SF, else DejaVu.
import json,pathlib,math,subprocess,textwrap,sys
from PIL import Image,ImageDraw,ImageFont
cmp_path=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);title=sys.argv[3]
sets=json.loads(cmp_path.read_text());n=len(sets);fps=12;W=1280;H=200+600*n
import os
def pick(env,*paths):
 for p in [os.environ.get(env)]+list(paths):
  if p and os.path.exists(p):return p
 raise SystemExit(f'no font found; set {env}')
font=pick('RENDER_FONT','/System/Library/Fonts/SFNS.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
mono=pick('RENDER_MONO','/System/Library/Fonts/SFNSMono.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
def f(s,m=False):return ImageFont.truetype(mono if m else font,s)
F={s:f(s) for s in [18,20,22,25,30,38]};M=f(17,True)
COLORS=['#5fe0ae','#92a8c9','#e0b35f','#c992c9']
end=max(d['capture']['wall_s'] for d in sets)+4
preview='--preview-only' in sys.argv
marks=[0,360,864,math.ceil(end*fps)-1]
proc=None if preview else subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
for frame in (marks if preview else range(math.ceil(end*fps))):
 t=frame/fps;im=Image.new('RGB',(W,H),'#080e17');d=ImageDraw.Draw(im)
 d.text((30,20),title,font=F[38],fill='#f3f6fb')
 d.text((30,70),'2 × DGX Spark · ~16K input per request · 400 output tokens · arrivals at 0 / 1 / 2 / 3 seconds',font=F[22],fill='#a8b9cb')
 d.text((1030,29),f'{t:05.1f}s  /  1×',font=F[25],fill='#a8b9cb')
 for c,item in enumerate(sets):
  a=item['capture'];x=30;oy=c*600;color=COLORS[c]
  d.text((x,115+oy),item['title'],font=F[30],fill=color)
  d.text((x,155+oy),item['subtitle'],font=F[20],fill='#bdcad8')
  d.text((x,183+oy),item['detail'],font=F[18],fill='#8296ac')
  delivered=0
  for j,l in enumerate(a['requests']):
   px=x+(j%2)*615;y=220+oy+(j//2)*215;done=t>=l['ended_s'];ev=[e for e in l['events'] if e['t']<=t]
   count=sum(e.get('tokens',0) for e in ev);count=l['completion_tokens'] if done else min(count,l['completion_tokens']);delivered+=count
   state='DONE' if done else ('STREAMING' if l['first_token_s'] is not None and t>=l['first_token_s'] else ('PREFILL / WAIT' if t>=l['scheduled_s'] else 'NOT SENT'))
   d.rounded_rectangle((px,y,px+606,y+195),radius=10,fill='#102b26' if done else '#111d2b',outline='#315647' if done else '#273b51')
   d.text((px+14,y+10),f'{j+1:02d}  {l["label"]}',font=F[22],fill=color);d.text((px+337,y+13),f'{state}  {count}/400',font=F[18],fill='#d5e5ee')
   txt=''.join(e['text'] for e in ev).replace('\n',' ');lines=textwrap.wrap(txt,width=50)[-4:]
   for k,line in enumerate(lines):d.text((px+14,y+45+k*23),line,font=M,fill='#cddbe8')
   d.rectangle((px+14,y+175,px+592,y+180),fill='#29394a');d.rectangle((px+14,y+175,px+14+578*count/400,y+180),fill=color)
  finished=t>=a['wall_s'];d.text((x,650+oy),f'{delivered:,} / 1,600 tokens delivered',font=F[25],fill=color)
  d.text((x+650,650+oy),f'Finished in {a["wall_s"]:.2f}s' if finished else 'Processing measured stream events…',font=F[25],fill='#f3f6fb')
 d.text((30,H-80),'Median of 3 runs per recipe. Recorded separately on the same pair; serving configurations differ.',font=F[20],fill='#a8b9cb')
 d.text((30,H-45),'Real SSE arrival times. Token animation uses server totals distributed across text deltas. Full elapsed time retained.',font=F[18],fill='#8296ac')
 if frame in marks:im.save(out.with_name(out.stem+f'-preview-{frame}.png'))
 if proc:proc.stdin.write(im.tobytes())
if proc:
 proc.stdin.close();assert proc.wait()==0
print(out)
