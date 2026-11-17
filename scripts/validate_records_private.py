#!/usr/bin/env python3
"""Validate submitted stored A/B records, source IDs, dates and provenance. Standard library only."""
from pathlib import Path
import csv, json, sys, collections, re, datetime
BASE=Path(__file__).resolve().parent.parent
RUN=BASE/'06_Experiment_Records'
RES=BASE/'03_Results'
QIDS=['H01','H02','H03','H04','H05','H06','H07','H09','H10','H11','H12']
PATHS={'USER_PASTED_B_RESPONSE':'B_User_Pasted_Transcribed','AUTHOR_CONFIRMATION_ONLY':'B_Author_Confirmed_Reconstructed','ASSISTANT_IN_CHAT_ASSESSMENT':'B_InChat_Assessment'}
def fail(msg):raise ValueError(msg)
def main():
  prov={r['question_id']:r for r in csv.DictReader((RUN/'B_OUTPUT_PROVENANCE.csv').open(encoding='utf8'))}
  labels=list(csv.DictReader((RES/'claim_annotations_PROVISIONAL.csv').open(encoding='utf8')))
  assert len(labels)==85 and len(prov)==11
  by_q=collections.defaultdict(list)
  for r in labels:by_q[r['question_id']].append(r)
  n_cites=n_claims=n_pasted=0
  cutoff=datetime.datetime.fromisoformat('2025-06-30T23:59:59+00:00')
  promptbase=BASE/'07_Prompt_And_Context_Evidence'
  for q in QIDS:
    A=json.loads((RUN/'A_Baseline_Transcribed'/f'{q}.json').read_text())
    cs=A['claims'];n_claims+=len(cs)
    assert A['abstention'] is False and len(cs)==len(by_q[q])
    ids=[c['id'] for c in cs]
    assert ids==[f'C{i:02d}' for i in range(1,len(ids)+1)]
    if (promptbase/'Prompts_Original_Seven'/f'{q}_A_PROMPT.txt').exists():
      ctx=(promptbase/'Prompts_Original_Seven'/f'{q}_A_PROMPT.txt').read_text()
    else:
      ctx=(promptbase/'Additional_Repaired_Candidate_Contexts'/f'{q}_REVIEW_CANDIDATE_NOT_EXACT_RUN.txt').read_text()
    known_src=set(re.findall(r'(?m)^SOURCE: (\S+)$',ctx))
    known_dates=re.findall(r'(?m)^DATE: (\d{4}-\d{2}-\d{2}T[^\n]+)$',ctx)
    if len(known_src)!=8 or len(known_dates)!=8 or len(set(known_src))!=8:fail(q+' source context must contain exactly 8 unique source IDs')
    historical=[datetime.datetime.fromisoformat(v.replace('Z','+00:00'))<=cutoff for v in known_dates]
    if historical.count(True)!=4 or historical.count(False)!=4:fail(q+' context not 4 PRE + 4 POST')
    for cl,la in zip(cs,by_q[q]):
      if la['claim_id']!=cl['id'] or la['claim_text']!=cl['text'] or json.loads(la['citations_json'])!=cl['citations']:fail(q+' mismatch claim '+cl['id'])
      if la['citation_ids_exist']!='True' or la['cited_versions_all_pre_cutoff']!='True':fail(q+' invalid citation/date')
      if any(sid not in known_src for sid in cl['citations']):fail(q+' cited SOURCE missing from question context')
      cited_dates=json.loads(la['cited_source_dates_json'])
      if len(cited_dates)!=len(cl['citations']) or not all(datetime.datetime.fromisoformat(v.replace('Z','+00:00'))<=cutoff for v in cited_dates):
        fail(q+' source timestamp after cutoff')
      n_cites+=len(cl['citations'])
    origin=prov[q]['B_decision_origin']
    B=json.loads((RUN/PATHS[origin]/f'{q}.json').read_text())
    assert set(B)=={'retain_claim_ids','decisions'} and set(B['decisions'])==set(ids)
    assert set(B['decisions'].values())<=set(['KEEP','DROP_POST_CUTOFF','DROP_UNSUPPORTED','DROP_UNCERTAIN'])
    assert B['retain_claim_ids']==[i for i in ids if B['decisions'][i]=='KEEP']
    if origin=='USER_PASTED_B_RESPONSE':n_pasted+=1
  assert n_claims==85 and n_cites==86 and n_pasted==4
  print('PASS: 11 A, 85 claims, 86 pre-cutoff citation links, B guards, 4 chat-pasted B transcripts')
  print('WARNING: remaining 7 B decisions are author confirmations or same-conversation assessments, NOT independent raw runs')
  print('WARNING: 85 support labels are SINGLE-ASSISTANT PROVISIONAL, NOT independent human gold')
if __name__=='__main__':
  try:main()
  except Exception as e:print('ERROR:',str(e),file=sys.stderr);raise
