#!/usr/bin/env python3
"""Recompute E2v4 five-arm citation-DATE results from saved raw/canonical records.

Uses the previously documented seven-run selection manifest; does not choose runs
using outcomes. Includes H02 earliest-completion sensitivity analysis. No model calls.
No factual-support or independent gold accuracy claims are computed.
"""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter
from pathlib import Path
from run_experiment import ROOT, CUTOFF, validate_a, validate_b, rule_date_only
from datetime import datetime

E2=ROOT/'12_Experiment2_Challenge'
RUNS=E2/'runs_v4'
SELECTION=E2/'E2v4_primary_run_manifest.csv'
FIELDS=['case','run_folder','arm','claims','pre_only','post_only','mixed','any_post','kept','deleted','date_correct_drop','date_false_drop','date_post_kept','other_reason_drop']
ARMS=['A_strict','A0_weak','A0_plus_rule','A_plus_Qwen','A0_plus_Qwen']

def get(p):
    return json.loads(p.read_text(encoding='utf-8'))

def get_payload(folder):
    meta=get(folder/'metadata.json')
    complete=get(folder/'COMPLETED.json')
    assert complete['status']=='COMPLETE_VALID_FORMAT'
    assert meta['protocol_version']=='E2V4_OLLAMA_NATIVE_SCHEMA_ENFORCED'
    assert meta['seed_requested']==42 and meta['model_requested']=='atr-qwen3b'
    assert meta['temperature_requested']==0.0
    context=meta['context']
    a=validate_a(get(folder/'strict_parsed.json'),context)
    a0=validate_a(get(folder/'weak_parsed.json'),context)
    rule=validate_b(get(folder/'date_rule_decisions.json'),a0)
    b=validate_b(get(folder/'strict_llm_verifier_decisions.json'),a)
    b0=validate_b(get(folder/'weak_llm_verifier_decisions.json'),a0)
    assert rule == rule_date_only(a0,context), 'Date rule differs from deterministic result'
    return meta,context,{'A_strict':(a,None),'A0_weak':(a0,None),'A0_plus_rule':(a0,rule),'A_plus_Qwen':(a,b),'A0_plus_Qwen':(a0,b0)}

def summarize(folder):
    meta,ctx,arms=get_payload(folder)
    dt_by_source={k:datetime.fromisoformat(v) for k,v in ctx['source_dates'].items()}
    rows=[]
    for arm in ARMS:
        a,b=arms[arm]
        d=dict(case=meta['qid'],run_folder=folder.name,arm=arm,claims=len(a['claims']),pre_only=0,post_only=0,mixed=0,any_post=0,kept=0,deleted=0,date_correct_drop=0,date_false_drop=0,date_post_kept=0,other_reason_drop=0)
        for c in a['claims']:
            date_flags=[dt_by_source[s]>CUTOFF for s in c['citations']]
            post=any(date_flags)
            typ='mixed' if post and not all(date_flags) else ('post_only' if post else 'pre_only')
            d[typ]+=1
            d['any_post']+=int(post)
            dec='KEEP' if b is None else b['decisions'][c['id']]
            if dec=='KEEP':
                d['kept']+=1
                if post:d['date_post_kept']+=1
            else:
                d['deleted']+=1
                if dec=='DROP_POST_CUTOFF':
                    d['date_correct_drop' if post else 'date_false_drop']+=1
                else:d['other_reason_drop']+=1
        assert d['kept']+d['deleted']==d['claims']
        rows.append(d)
    return rows

def write_csv(p,rows,fields):
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

selection=list(csv.DictReader(SELECTION.open(encoding='utf8')))
assert len(selection)==7 and len({r['question_id'] for r in selection})==7
primary=[]
for r in selection:
    folder=RUNS/r['run_folder']
    assert folder.exists(), f'Missing selected run {folder}'
    rows=summarize(folder)
    assert rows[0]['case']==r['question_id']
    primary.extend(rows)
write_csv(E2/'primary_five_arm_comparison_VERIFIED.csv',primary,FIELDS)
summary={arm:{k:sum(x[k] for x in primary if x['arm']==arm) for k in FIELDS[3:]} for arm in ARMS}

h02_selected=next(r['run_folder'] for r in selection if r['question_id']=='H02')
extras=sorted(p.parent for p in RUNS.glob('*/COMPLETED.json') if p.parent.name not in {r['run_folder'] for r in selection})
assert len(extras)==1 and get(extras[0]/'metadata.json')['qid']=='H02'
assert get(extras[0]/'metadata.json')['created_utc'] < get((RUNS/h02_selected)/'metadata.json')['created_utc']
alternative=[r for r in primary if r['case']!='H02']+summarize(extras[0])
alt_summary={arm:{k:sum(x[k] for x in alternative if x['arm']==arm) for k in FIELDS[3:]} for arm in ARMS}
sensitivity=[]
for arm in ARMS:
    row={'arm':arm}
    for k in ('claims','any_post','kept','deleted','date_correct_drop','date_false_drop','date_post_kept'):
        row['reported_H02_'+k]=summary[arm][k];row['earliest_H02_'+k]=alt_summary[arm][k]
    sensitivity.append(row)
write_csv(E2/'H02_earliest_run_sensitivity.csv',sensitivity,list(sensitivity[0]))

summary_obj={'selected_rationale':'first researcher-reported E2v4 completion for each case, NOT first chronological completion for H02',
 'selected_run_ids':{r['question_id']:r['run_folder'] for r in selection},
 'excluded_earlier_h02':extras[0].name,'limits':['Citation-date checks only, no source support or gold-fact annotation','Model seed requested, exact determinism not established','Exploratory post-hoc selection and duplicate-run sensitivity disclosed'],
 'primary':summary,'earliest_H02_substitution':alt_summary}
(E2/'recomputed_primary_summary.json').write_text(json.dumps(summary_obj,indent=2)+'\n',encoding='utf8')
print('Seven selected runs independently recomputed from model outputs (five arms per case).')
for arm, vals in summary.items(): print(arm, 'claims',vals['claims'],'any_post',vals['any_post'],'kept',vals['kept'],'correct_drop',vals['date_correct_drop'],'false_drop',vals['date_false_drop'],'post_kept',vals['date_post_kept'])
print('EARLIEST H02 instead of researcher-reported H02:')
for arm, vals in alt_summary.items(): print(arm, 'claims',vals['claims'],'any_post',vals['any_post'],'kept',vals['kept'],'correct_drop',vals['date_correct_drop'],'false_drop',vals['date_false_drop'],'post_kept',vals['date_post_kept'])
