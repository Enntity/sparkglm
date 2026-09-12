# SPDX-License-Identifier: AGPL-3.0-only
"""Independent synthetic fixtures for the generic worker artifact."""
import copy,importlib.util,json,random,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('route_helper',Path(__file__).with_name('original.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def balanced():
    return [{'group':'g','owner':str(o),'layer':0,'routes':[[o,8+o],[4+o,12+o]]} for o in range(4)]
def skew_or_shared(shared=False):
    records=[];local=remote=0
    for o in range(4):
        if shared:ids=[0,1,2,3,32,33,34,35]
        else:
            a,b=(7,1) if o%2==0 else (1,7)
            ids=list(range(local,local+a))+list(range(32+remote,32+remote+b));local+=a;remote+=b
        records.append({'group':'g','owner':str(o),'layer':0,'routes':[ids[:],ids[:],ids[:]]})
    return records

def layer(result):return result['groups'][0]['layers'][0]

class RouteTests(unittest.TestCase):
    def invoke(self,records,expert_count=16,split=8,owners=4,layers=1):
        before=copy.deepcopy(records)
        result=helper.analyze_routes(records,expert_count,split,owners,layers)
        self.assertEqual(records,before);json.dumps(result,allow_nan=False)
        self.assertEqual(result['metric_scope'],'expert_counts_not_latency_prediction')
        return result

    def test_balanced_no_reuse_exact_counts(self):
        r=self.invoke(balanced());x=layer(r)
        self.assertEqual(r['geometry'],{'rows':2,'topk':2,'expert_count':16,'split':8,'expected_owners':4,'expected_layers':1})
        self.assertEqual(x,{'layer':0,'per_owner':[{'owner':str(o),'rank_unique':[2,2]} for o in range(4)],
            'serial_critical_expert_visits':8,'joint_rank_unique':[8,8],'joint_critical_experts':8,
            'owner_unique_visits':16,'joint_unique_experts':16,'reuse_factor':1.0,'critical_count_ratio':1.0})

    def test_skew_balance_gain_without_cross_owner_reuse(self):
        x=layer(self.invoke(skew_or_shared(),64,32))
        self.assertEqual([r['rank_unique'] for r in x['per_owner']],[[7,1],[1,7],[7,1],[1,7]])
        self.assertEqual(x['joint_rank_unique'],[16,16]);self.assertEqual(x['serial_critical_expert_visits'],28)
        self.assertEqual(x['critical_count_ratio'],1.75);self.assertEqual(x['reuse_factor'],1.0)

    def test_shared_experts_reuse(self):
        x=layer(self.invoke(skew_or_shared(True),64,32))
        self.assertEqual(x['joint_rank_unique'],[4,4]);self.assertEqual(x['serial_critical_expert_visits'],16)
        self.assertEqual(x['owner_unique_visits'],32);self.assertEqual(x['joint_unique_experts'],8)
        self.assertEqual(x['critical_count_ratio'],4.0);self.assertEqual(x['reuse_factor'],4.0)

    def test_permutations_multigroup_multilayer_and_immutability(self):
        records=[]
        for g in ['z','a']:
            for l in [1,0]:
                for r in balanced():r.update(group=g,layer=l);records.append(r)
        expected=self.invoke(records,layers=2);rng=random.Random(4)
        for _ in range(10):
            changed=copy.deepcopy(records);rng.shuffle(changed)
            for r in changed:
                rng.shuffle(r['routes'])
                for row in r['routes']:rng.shuffle(row)
            self.assertEqual(self.invoke(changed,layers=2),expected)
        self.assertEqual([g['group'] for g in expected['groups']],['a','z'])
        self.assertEqual([l['layer'] for l in expected['groups'][0]['layers']],[0,1])

    def test_rank_boundaries(self):
        for split,wanted in [(0,[0,16]),(16,[16,0])]:
            x=layer(self.invoke(balanced(),split=split));self.assertEqual(x['joint_rank_unique'],wanted)
            self.assertEqual(x['critical_count_ratio'],1.0)
        one=[{'group':'g','owner':'only','layer':0,'routes':[[0]]}]
        self.assertEqual(layer(self.invoke(one,1,0,1))['reuse_factor'],1.0)

    def test_reject_incomplete_duplicate_and_inconsistent_owners(self):
        cases=[balanced()[:-1],balanced()+[copy.deepcopy(balanced()[0])]]
        extra=balanced();extra.append({'group':'g','owner':'extra','layer':0,'routes':[[0,1],[2,3]]});cases.append(extra)
        for records in cases:
            before=copy.deepcopy(records)
            with self.assertRaises(ValueError):helper.analyze_routes(records,16,8,4,1)
            self.assertEqual(records,before)
        records=[{'group':'g','owner':'a','layer':0,'routes':[[0]]},{'group':'g','owner':'b','layer':1,'routes':[[0]]}]
        with self.assertRaises(ValueError):helper.analyze_routes(records,2,1,2,2)

    def test_malformed_records_rejected_without_mutation(self):
        base=balanced();cases=[None,(),{},[],[1]]
        edits=[lambda r:r.update(group=''),lambda r:r.update(group=3),lambda r:r.update(owner=False),
               lambda r:r.update(owner=''),lambda r:r.update(layer=True),lambda r:r.update(layer=1),
               lambda r:r.update(layer=-1),lambda r:r.update(layer=0.0),lambda r:r.update(routes=[]),
               lambda r:r.update(routes=()),lambda r:r.update(routes=[[]]),lambda r:r.update(routes=[[0,0],[1,2]]),
               lambda r:r.update(routes=[[True,1],[2,3]]),lambda r:r.update(routes=[[-1,1],[2,3]]),
               lambda r:r.update(routes=[[16,1],[2,3]]),lambda r:r.update(routes=[[0.0,1],[2,3]]),
               lambda r:r.update(routes=[[0,1],[2]]),lambda r:r.update(routes=[[0,1]]),lambda r:r.update(routes=[[0],[1]]),
               lambda r:r.update(routes=[[0,1],(2,3)]),lambda r:r.update(extra='unexpected'),lambda r:r.pop('owner')]
        for edit in edits:
            rows=copy.deepcopy(base);edit(rows[0]);cases.append(rows)
        for rows in cases:
            with self.subTest(records=rows):
                before=copy.deepcopy(rows)
                with self.assertRaises(ValueError):helper.analyze_routes(rows,16,8,4,1)
                self.assertEqual(rows,before)

    def test_invalid_configuration(self):
        args=[16,8,4,1]
        for index,values in [(0,[0,-1,True,16.0,None]),(1,[-1,17,True,8.0,None]),(2,[0,-1,True,4.0,None]),(3,[0,-1,True,1.0,None])]:
            for value in values:
                changed=args[:];changed[index]=value
                with self.subTest(index=index,value=value),self.assertRaises(ValueError):helper.analyze_routes(balanced(),*changed)

    def test_random_small_unique_set_reference(self):
        rng=random.Random(31)
        for _ in range(80):
            n=rng.randint(1,16);split=rng.randint(0,n);owners=rng.randint(1,5);nr=rng.randint(1,4);k=rng.randint(1,n)
            records=[{'group':'g','owner':str(o),'layer':0,'routes':[rng.sample(range(n),k) for _ in range(nr)]} for o in range(owners)]
            x=layer(self.invoke(records,n,split,owners));owner_sets=[set(sum(r['routes'],[])) for r in records]
            union=set().union(*owner_sets);joint=[sum(e<split for e in union),sum(e>=split for e in union)]
            serial=sum(max(sum(e<split for e in s),sum(e>=split for e in s)) for s in owner_sets)
            self.assertEqual(x['joint_rank_unique'],joint);self.assertEqual(x['serial_critical_expert_visits'],serial)
            self.assertEqual(x['reuse_factor'],sum(map(len,owner_sets))/len(union));self.assertEqual(x['critical_count_ratio'],serial/max(joint))

if __name__=='__main__':unittest.main()
