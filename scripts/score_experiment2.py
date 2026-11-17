#!/usr/bin/env python3
"""Report citation-DATE structure for genuine E2 run folders; NOT factual support.
Run: python3 scripts/score_experiment2.py
"""
from __future__ import annotations
import csv
import datetime as dt
import json
from collections import Counter
from html import escape
from pathlib import Path
from run_experiment import ROOT, OUTDIR, CUTOFF, validate_a, validate_b, rule_date_only


def metrics(claims, source_dates):
    dates={k:dt.datetime.fromisoformat(v) for k,v in source_dates.items()}
    post=0;pre=0;mixed=0;unknown=0;uncited=0;links=0
    for c in claims:
        cites=c['citations'];links+=len(cites)
        if not cites:uncited+=1;continue
        if any(x not in dates for x in cites): unknown+=1;continue
        flags=[dates[x]>CUTOFF for x in cites]
        if all(flags):post+=1
        elif any(flags):mixed+=1
        else:pre+=1
    return {'claim_records':len(claims),'citation_links':links,'pre_only_cited':pre,
            'post_only_cited':post,'mixed_date_cited':mixed,
            'unknown_cited':unknown,'uncited':uncited,'any_post_cited':post+mixed}


def main():
    runs=sorted(OUTDIR.glob('*/COMPLETED.json')) if OUTDIR.exists() else []
    if not runs:
        print('NOT RUN: no complete Experiment 2 model outputs; nothing to score.')
        return
    rows=[]
    reason_counts={"date_rule_B":Counter(),"A0_plus_llm_B":Counter(),"A_plus_llm_B":Counter()}
    for marker in runs:
        folder=marker.parent
        meta=json.loads((folder/'metadata.json').read_text())
        if meta['study']!='E2_POST_FIRST_EXTENSION_NOT_E1_RERUN' or meta.get('protocol_version')!='E2V4_OLLAMA_NATIVE_SCHEMA_ENFORCED':raise ValueError('Wrong study type or protocol version')
        ctx=meta['context']
        strict=validate_a(json.loads((folder/'strict_parsed.json').read_text()),ctx)
        weak=validate_a(json.loads((folder/'weak_parsed.json').read_text()),ctx)
        rule=json.loads((folder/'date_rule_decisions.json').read_text())
        bweak=json.loads((folder/'weak_llm_verifier_decisions.json').read_text())
        bstrict=json.loads((folder/'strict_llm_verifier_decisions.json').read_text())
        validate_b(rule,weak);validate_b(bweak,weak);validate_b(bstrict,strict)
        reason_counts['date_rule_B'].update(v for v in rule['decisions'].values() if v!='KEEP')
        reason_counts['A0_plus_llm_B'].update(v for v in bweak['decisions'].values() if v!='KEEP')
        reason_counts['A_plus_llm_B'].update(v for v in bstrict['decisions'].values() if v!='KEEP')
        if rule!=rule_date_only(weak,ctx):raise ValueError('Rule mismatch '+str(folder))
        for arm,original,decisions in (
             ('A_strict',strict,None), ('A_plus_llm_B',strict,bstrict), ('A0_weak',weak,None),
             ('A0_plus_date_rule_B',weak,rule), ('A0_plus_llm_B',weak,bweak)):
            claims=(original['claims'] if decisions is None else
                    [c for c in original['claims'] if c['id'] in decisions['retain_claim_ids']])
            record={'run':folder.name,'question_id':meta['qid'],'repeat':meta['repeat'],'arm':arm,
                    'model_requested':meta['model_requested'],'seed_requested':meta['seed_requested'],
                    'protocol_version':meta.get('protocol_version','E2V2'),
                    'format_warnings':'; '.join(json.loads((folder/'COMPLETED.json').read_text()).get('format_warnings',[])),
                    'temperature_requested':meta['temperature_requested'],
                    'retained_claims':len(claims),
                    'deleted_claims':0 if decisions is None else len(original['claims'])-len(claims)}
            record.update(metrics(claims,ctx['source_dates']))
            record['factual_support_accuracy']='NA_NO_INDEPENDENT_LABELS'
            record['gold_coverage']='NA_NO_GOLD_MAPPING'
            record['citations_are_structural_not_support']='YES'
            rows.append(record)
    out=ROOT/'12_Experiment2_Challenge/structural_results_v4.csv'
    with out.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('E2 genuine complete runs:',len(runs),'| structural rows:',len(rows))
    # Figure 2: counts by recorded decision reasons, generated ONLY after real complete runs.
    fig=ROOT/'12_Experiment2_Challenge/E2v4_drops_by_reported_reason.svg'
    labels=['DROP_POST_CUTOFF','DROP_UNSUPPORTED','DROP_UNCERTAIN']
    max_n=max([1]+[reason_counts[arm][label] for arm in reason_counts for label in labels])
    parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 395" role="img" aria-label="Recorded dropped claim counts by reason from genuine Experiment 2 runs">',
           '<rect width="900" height="395" fill="white"/>',
           '<text x="30" y="36" font-family="Arial" font-size="21">E2 dropped claim records by recorded reason</text>',
           '<text x="30" y="59" font-family="Arial" font-size="12">Structural counts; LLM reasons are not independently adjudicated</text>']
    width=515
    for i,label in enumerate(labels):
        y=107+i*89
        parts.append(f'<text x="30" y="{y+16}" font-family="Arial" font-size="13">{escape(label)}</text>')
        for j,(arm,color) in enumerate([('date_rule_B','#6b7280'),('A0_plus_llm_B','#2563eb'),('A_plus_llm_B','#059669')]):
            count=reason_counts[arm][label]
            bar_y=y+17+j*18
            px=count*width/max_n
            parts.append(f'<rect x="260" y="{bar_y}" width="{px:.1f}" height="15" fill="{color}"/>')
            parts.append(f'<text x="{270+px:.1f}" y="{bar_y+13}" font-family="Arial" font-size="13">{count}</text>')
    parts.extend(['<rect x="30" y="370" width="12" height="12" fill="#6b7280"/>',
        '<text x="48" y="381" font-family="Arial" font-size="12">Date-only on A0</text>',
        '<rect x="230" y="370" width="12" height="12" fill="#2563eb"/>',
        '<text x="248" y="381" font-family="Arial" font-size="12">LLM B on A0</text>',
        '<rect x="410" y="370" width="12" height="12" fill="#059669"/>',
        '<text x="428" y="381" font-family="Arial" font-size="12">LLM B on strict A</text>', '</svg>'])
    fig.write_text('\n'.join(parts),encoding='utf-8')
    print('Saved',out,'and',fig)
    print('CAUTION: these are date/citation STRUCTURAL metrics, not true R_post, citation precision or claim-support accuracy.')

if __name__=='__main__':main()
