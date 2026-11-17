#!/usr/bin/env python3
"""Score curated fixed-context A/B claim records, with provenance and NA handling.

Only scores available/annotated claims; does not call LLM APIs or manufacture
independent B outputs. The command does not change raw JSON records.

python 05_Code_and_Environment/07_Claim_Evaluation/score_claims.py
"""
import csv,json,pathlib,collections
ROOT=pathlib.Path(__file__).resolve().parent.parent
RUN=ROOT/'06_Experiment_Records';OUT=ROOT/'03_Results'
QIDS=['H01','H02','H03','H04','H05','H06','H07','H09','H10','H11','H12']
ALLOWED={'KEEP','DROP_POST_CUTOFF','DROP_UNSUPPORTED','DROP_UNCERTAIN'}

def rate(n,d):return round(n/d,6) if d else None

def readcsv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

labels=readcsv(OUT/'claim_annotations_PROVISIONAL.csv')
ann={(r['question_id'],r['claim_id']):r for r in labels}
assert len(ann)==85 and len(labels)==85,'Expect distinct 85 annotated original claim records'
prov={r['question_id']:r for r in readcsv(RUN/'B_OUTPUT_PROVENANCE.csv')}
assert set(prov)==set(QIDS)
rows=[];bib=[]
for q in QIDS:
    A=json.loads((RUN/'A_Baseline_Transcribed'/f'{q}.json').read_text(encoding='utf-8'))
    assert set(A)=={'claims','abstention'} and not A['abstention']
    a_ids=[c['id'] for c in A['claims']]
    assert a_ids==[f'C{i:02d}' for i in range(1,len(a_ids)+1)]
    origin=prov[q]['B_decision_origin']
    dirname={'USER_PASTED_B_RESPONSE':'B_User_Pasted_Transcribed','AUTHOR_CONFIRMATION_ONLY':'B_Author_Confirmed_Reconstructed','ASSISTANT_IN_CHAT_ASSESSMENT':'B_InChat_Assessment'}[origin]
    B=json.loads((RUN/dirname/f'{q}.json').read_text(encoding='utf-8'))
    assert set(B)=={'retain_claim_ids','decisions'}
    assert set(B['decisions'])==set(a_ids)
    assert all(d in ALLOWED for d in B['decisions'].values())
    keep=[i for i in a_ids if B['decisions'][i]=='KEEP']
    assert B['retain_claim_ids']==keep,'B subsequence integrity violation'
    a_support=[ann[q,i]['support_label'] for i in a_ids]
    a_date=[ann[q,i]['temporal_label'] for i in a_ids]
    for i in a_ids:
        r=ann[q,i]
        assert r['citation_ids_exist']=='True' and r['cited_versions_all_pre_cutoff']=='True'
        assert r['support_label'] in ('SUPPORTED','UNSUPPORTED','UNCERTAIN')
        assert r['temporal_label'] in ('PRE_ONLY','POST_ONLY','MIXED','UNSUPPORTED','UNKNOWN')
    retained=[i for i in keep]
    b_support=[ann[q,i]['support_label'] for i in retained]
    b_date=[ann[q,i]['temporal_label'] for i in retained]
    A_sup=sum(x=='SUPPORTED' for x in a_support)
    B_sup=sum(x=='SUPPORTED' for x in b_support)
    A_post=sum(s=='SUPPORTED' and d=='POST_ONLY' for s,d in zip(a_support,a_date))
    B_post=sum(s=='SUPPORTED' and d=='POST_ONLY' for s,d in zip(b_support,b_date))
    A_unc=sum(x=='UNCERTAIN' for x in a_support)
    B_unc=sum(x=='UNCERTAIN' for x in b_support)
    A_unsp=sum(x=='UNSUPPORTED' for x in a_support)
    B_unsp=sum(x=='UNSUPPORTED' for x in b_support)
    eligible=[i for i in a_ids if ann[q,i]['support_label']=='SUPPORTED' and ann[q,i]['temporal_label']=='PRE_ONLY']
    deleted_eligible=[i for i in eligible if i not in keep]
    rows.append({'question_id':q,'B_provenance':origin,'A_claim_records':len(a_ids),'B_kept':len(keep),'B_dropped':len(a_ids)-len(keep),
         'A_supported_provisional':A_sup,'B_supported_provisional':B_sup,
         'A_unsupported_provisional':A_unsp,'B_unsupported_provisional':B_unsp,
         'A_uncertain_provisional':A_unc,'B_uncertain_provisional':B_unc,
         'A_POST_ONLY_supported':A_post,'B_POST_ONLY_supported':B_post,
         'A_primary_post_only_rate':rate(A_post,A_sup),'B_primary_post_only_rate':rate(B_post,B_sup),
         'A_unsupported_rate_excl_uncertain':rate(A_unsp,len(a_ids)-A_unc),
         'B_unsupported_rate_excl_uncertain':rate(B_unsp,len(keep)-B_unc),
         'A_eligible_supported':len(eligible),'B_deleted_eligible':len(deleted_eligible),
         'overfiltering_rate':rate(len(deleted_eligible),len(eligible)),
         'gold_pre_fact_coverage':None,'support_review_status':'SINGLE_ASSISTANT_PROVISIONAL_NONBLIND'})

with (OUT/'results_per_question.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def sumkey(key):return sum(int(x[key]) for x in rows)
raw=[r for r in rows if r['B_provenance']=='USER_PASTED_B_RESPONSE']
summary={
    'analysis_status':'PROVISIONAL_SINGLE_ASSISTANT_SOURCE_SUPPORT_REVIEW_NOT_INDEPENDENT_GOLD',
    'study_scope':'curated fixed context 11 questions, 4 PRE plus 4 POST passages each',
    'historical_cutoff_utc':'2025-06-30T23:59:59Z',
    'heldout_questions':len(QIDS),'A_claim_records':sumkey('A_claim_records'),
    'citation_pairs_structurally_valid':sum(len(json.loads(r['citations_json'])) for r in labels),
    'A_supported_assistant_provisional':sumkey('A_supported_provisional'),
    'A_unsupported_assistant_provisional':sumkey('A_unsupported_provisional'),
    'A_uncertain_assistant_provisional':sumkey('A_uncertain_provisional'),
    'B_KEEP_mixed_provenance':sumkey('B_kept'),'B_DROP_mixed_provenance':sumkey('B_dropped'),
    'B_individually_user_pasted_json_pairs':len(raw),
    'B_author_confirmed_no_individual_raw_json_pairs':sum(r['B_provenance']=='AUTHOR_CONFIRMATION_ONLY' for r in rows),
    'B_assistant_inchat_not_independent_pairs':sum(r['B_provenance']=='ASSISTANT_IN_CHAT_ASSESSMENT' for r in rows),
    'A_post_only_supported':sumkey('A_POST_ONLY_supported'),
    'B_post_only_supported_mixed_provenance':sumkey('B_POST_ONLY_supported'),
    'A_primary_post_only_rate':rate(sumkey('A_POST_ONLY_supported'),sumkey('A_supported_provisional')),
    'B_primary_post_only_rate_mixed_provenance':rate(sumkey('B_POST_ONLY_supported'),sumkey('B_supported_provisional')),
    'A_primary_denominator':sumkey('A_supported_provisional'),
    'B_primary_denominator_mixed_provenance':sumkey('B_supported_provisional'),
    'absolute_risk_difference_B_minus_A':rate(sumkey('B_POST_ONLY_supported'),sumkey('B_supported_provisional'))-rate(sumkey('A_POST_ONLY_supported'),sumkey('A_supported_provisional')),
    'relative_improvement':None,
    'eligible_supported_claims_assistant_provisional':sumkey('A_eligible_supported'),
    'overfiltered_eligible_claims':sumkey('B_deleted_eligible'),
    'overfiltering_rate_mixed_provenance':rate(sumkey('B_deleted_eligible'),sumkey('A_eligible_supported')),
    'independent_gold_pre_cutoff_coverage':None,
    'independent_citation_pair_precision':None,
    'raw_B_only_subset':{
        'questions':[r['question_id'] for r in raw],
        'claim_records':sum(r['A_claim_records'] for r in raw),
        'kept':sum(r['B_kept'] for r in raw),
        'dropped':sum(r['B_dropped'] for r in raw)
    },
    'method_limits':['11 questions exploratory; zero baseline post-cutoff citations: floor/ceiling effect',
        'Only four B decisions explicitly user-pasted as separate JSONs',
        'Two B records are author-confirmed but original standalone JSON for each absent',
        'Five B decisions made in same assistant conversation, not independent held-out runs',
        'Cited passage support labels assessed by nonblinded assistant in same project; no independent human adjudication',
        'No post-hoc atomic splitting of compound claim sentences; gold fact coverage not available',
        'ChatGPT UI temperature, seed and exact model revision not observed',
        'No statistical significance or benefit conclusions supported']}
OUT.joinpath('summary_results_PROVISIONAL.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2))
