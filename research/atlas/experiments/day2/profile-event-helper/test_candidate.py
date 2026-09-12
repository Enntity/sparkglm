# SPDX-License-Identifier: AGPL-3.0-only
"""Independent elementary-interval oracle; candidate supplied only after source review."""
import copy,importlib.util,json,math,os,random,unittest
from pathlib import Path

def oracle(events,windows):
    output=[]
    for w in sorted(windows,key=lambda w:(w['start'],w['end'],w['id'])):
        clips=[(e['name'],max(e['start'],w['start']),min(e['end'],w['end'])) for e in events]
        clips=[(n,a,b) for n,a,b in clips if b>a]
        endpoints=sorted({w['start'],w['end']}|{x for _,a,b in clips for x in (a,b)})
        # Independent of a candidate's required sorted merge algorithm.
        covered=math.fsum(b-a for a,b in zip(endpoints,endpoints[1:]) if any(x<=a and y>=b for _,x,y in clips))
        names=sorted({n for n,_,_ in clips})
        output.append(dict(id=w['id'],start=w['start'],end=w['end'],duration=w['end']-w['start'],covered_duration=covered,
            uncovered_duration=w['end']-w['start']-covered,by_name=[dict(name=n,count=sum(k==n for k,_,_ in clips),duration=math.fsum(b-a for k,a,b in clips if k==n)) for n in names]))
    return {'windows':output}

def generated():
    rng=random.Random(62041)
    for _ in range(100):
        windows=[dict(id=str(i),start=i*15,end=i*15+10) for i in range(rng.randrange(6))]
        events=[]
        for i in range(rng.randrange(35)):
            a=rng.randrange(-10,90);events.append(dict(name=rng.choice(['alpha','beta','γ']),start=a,end=a+rng.randrange(30)))
        yield events,windows

candidate=None
if os.environ.get('PROFILE_EVENT_CANDIDATE'):
    path=Path(os.environ['PROFILE_EVENT_CANDIDATE']).resolve()
    spec=importlib.util.spec_from_file_location('reviewed_candidate',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    candidate=mod.summarize_events

class OracleTests(unittest.TestCase):
    def test_manual_overlap_clipping_and_duplicate(self):
        e=[dict(name='a',start=-2,end=4),dict(name='a',start=2,end=7),dict(name='b',start=5,end=12)]
        w=[dict(id='x',start=0,end=10)]
        r=oracle(e,w)['windows'][0]
        self.assertEqual(r['covered_duration'],10);self.assertEqual(r['by_name'],[dict(name='a',count=2,duration=9),dict(name='b',count=1,duration=5)])
    def test_seeded_oracle_invariants(self):
        for e,w in generated():
            for row in oracle(e,w)['windows']:
                self.assertLessEqual(row['covered_duration'],row['duration'])
                self.assertGreaterEqual(sum(n['duration'] for n in row['by_name']),row['covered_duration'])

@unittest.skipUnless(candidate,'Set PROFILE_EVENT_CANDIDATE only after source review')
class CandidateTests(unittest.TestCase):
    def check_case(self,e,w):
        before=copy.deepcopy((e,w));out=candidate(e,w)
        self.assertEqual((e,w),before);self.assertEqual(out,oracle(e,w));json.dumps(out,allow_nan=False)
        return out
    def test_fixed_boundaries_empty_duplicate_spanning(self):
        w=[dict(id='b',start=10,end=20),dict(id='a',start=0,end=10),dict(id='c',start=30,end=40)]
        e=[dict(name='same',start=-5,end=25),dict(name='same',start=-5,end=25),dict(name='touch',start=20,end=30),dict(name='zero',start=5,end=5)]
        for events,windows in [(e,w),([],w),(e,[]),([],[])]:self.check_case(events,windows)
    def test_seeded_random_and_permutations(self):
        rng=random.Random(811)
        for e,w in generated():
            original=self.check_case(e,w);rng.shuffle(e);rng.shuffle(w)
            self.assertEqual(self.check_case(e,w),original)
    def test_fractional(self):
        self.check_case([dict(name='a',start=-.5,end=.25),dict(name='a',start=0,end=.5)],[dict(id='w',start=0,end=.5)])
    def test_malformed_types_and_geometry(self):
        good_e=dict(name='a',start=0,end=1);good_w=dict(id='w',start=0,end=2)
        bad=[]
        for value in [None,{},(),True,'x']:bad.extend([(value,[good_w]),([good_e],value)])
        for field in ['name','start','end']:
            d=dict(good_e);d.pop(field);bad.append(([d],[good_w]))
        for field in ['id','start','end']:
            d=dict(good_w);d.pop(field);bad.append(([good_e],[d]))
        for badname in ['',None,False,1]:
            bad.extend([([dict(good_e,name=badname)],[good_w]),([good_e],[dict(good_w,id=badname)])])
        for v in [True,None,'1',float('nan'),float('inf'),-float('inf'),10**400]:
            for key in ['start','end']:
                bad.extend([([dict(good_e,**{key:v})],[good_w]),([good_e],[dict(good_w,**{key:v})])])
        bad.extend([([dict(good_e,extra=1)],[good_w]),([good_e],[dict(good_w,extra=1)]),([dict(good_e,end=-1)],[]),([], [dict(good_w,end=0)]),([], [dict(good_w,end=-1)]),([], [good_w,good_w]),([],[good_w,dict(id='z',start=1,end=3)]),([None],[good_w]),([good_e],[None])])
        for e,w in bad:
            before=repr((e,w))
            with self.assertRaises(ValueError,msg=before):candidate(e,w)
            self.assertEqual(repr((e,w)),before)
    def test_computed_overflow(self):
        for e,w in [([], [dict(id='w',start=-1e308,end=1e308)]),([dict(name='a',start=0,end=1e308)]*2,[dict(id='w',start=0,end=1e308)])]:
            with self.assertRaises(ValueError):candidate(e,w)

if __name__=='__main__':unittest.main()
