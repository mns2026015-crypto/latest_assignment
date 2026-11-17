#!/usr/bin/env python3
"""Recompute structural five-arm results from released claim-level citation-date records.

This code cannot assess factual claim support or independently adjudicated gold accuracy.
"""
from collections import defaultdict
from pathlib import Path
import csv

ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/'results/E2v4_citation_date_decisions.csv'
EXPECTED={'A_strict':(51,16,51,0,0,16),'A_plus_Qwen':(51,16,29,4,18,12),
          'A0_weak':(39,20,39,0,0,20),'A0_plus_rule':(39,20,19,20,0,0),
          'A0_plus_Qwen':(39,20,30,7,2,13)}

def summarize():
    with FILE.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==90, f'Expected 90 saved claims; saw {len(rows)}'
    assert len({(r['qid'],r['generator_arm'],r['claim_id']) for r in rows})==90
    arms=defaultdict(lambda: dict(claims=0,any_post=0,kept=0,correct_drop=0,false_drop=0,post_kept=0))
    for r in rows:
        assert r['generator_arm'] in ('strict','weak')
        post=r['any_post_date']=='1'
        assert r['any_post_date'] in ('0','1')
        dec=r['B_decision']
        assert dec in ('KEEP','DROP_POST_CUTOFF','DROP_UNSUPPORTED','DROP_UNCERTAIN')
        name='A_strict' if r['generator_arm']=='strict' else 'A0_weak'
        model='A_plus_Qwen' if r['generator_arm']=='strict' else 'A0_plus_Qwen'
        names=[name,model]
        if r['generator_arm']=='weak': names.append('A0_plus_rule')
        for key in names:
            a=arms[key]
            a['claims']+=1;a['any_post']+=int(post)
            retained=(key==name) or (key=='A0_plus_rule' and not post) or (key==model and dec=='KEEP')
            a['kept']+=int(retained)
            a['post_kept']+=int(retained and post)
            if not retained and post and ((key=='A0_plus_rule') or dec=='DROP_POST_CUTOFF'):a['correct_drop']+=1
            if not retained and not post and key==model and dec=='DROP_POST_CUTOFF':a['false_drop']+=1
    return {k:tuple(arms[k][field] for field in ('claims','any_post','kept','correct_drop','false_drop','post_kept')) for k in EXPECTED}

if __name__=='__main__':
    summary=summarize()
    for arm,vals in summary.items():
        print(f'{arm:16s}: claims={vals[0]}, post={vals[1]}, kept={vals[2]}, correct_drops={vals[3]}, false_date_drops={vals[4]}, post_kept={vals[5]}')
    assert summary==EXPECTED, f'Discrepancy: {summary}'
    print('PASS: public citation-date aggregates match archived seven-run summary; claim support NOT evaluated')
