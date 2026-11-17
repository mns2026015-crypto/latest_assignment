#!/usr/bin/env python3
"""Experiment 2: independent stateless model calls; no calls in --dry-run.

Supports hosted Gemini/OpenAI-compatible APIs and optional local Ollama.
No model or experiment outcomes are packaged. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / '07_Prompt_And_Context_Evidence/Prompts_Original_Seven'
OUTDIR = ROOT / '12_Experiment2_Challenge/runs_v4'
CUTOFF = dt.datetime(2025, 6, 30, 23, 59, 59, tzinfo=dt.timezone.utc)
ID_PATTERN = re.compile(r'C\d\d')
SOURCES = re.compile(r'(?m)^SOURCE: (\S+)$')
ORIGINAL = re.compile(r'(?s)QUESTION:\n(.+?)\n\nEVIDENCE \(unchanged fixed order\):\n(.*)\Z')
QIDS = ('H01', 'H02', 'H03', 'H06', 'H07', 'H09', 'H12')
DECISIONS = {'KEEP', 'DROP_POST_CUTOFF', 'DROP_UNSUPPORTED', 'DROP_UNCERTAIN'}
PROTOCOL_VERSION = 'E2V4_OLLAMA_NATIVE_SCHEMA_ENFORCED'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(data: str):
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:  # never overwrite saved runs
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def source_blocks(text: str):
    starts = list(SOURCES.finditer(text))
    if len(starts) != 8:
        raise ValueError(f'Expected 8 source blocks, found {len(starts)}')
    blocks = []
    for i, match in enumerate(starts):
        segment = text[match.start():(starts[i+1].start() if i+1<len(starts) else len(text))].strip()
        date_match = re.search(r'(?m)^DATE: (\S+)$', segment)
        if not date_match:
            raise ValueError(f'Missing DATE: {match.group(1)}')
        stamp = dt.datetime.fromisoformat(date_match.group(1).replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            raise ValueError('Source date must include timezone')
        blocks.append({'id': match.group(1), 'date':stamp, 'text':segment})
    if len({b['id'] for b in blocks})!=8:
        raise ValueError('Duplicate source ID')
    if sum(b['date']<=CUTOFF for b in blocks)!=4:
        raise ValueError('Source mix is not four pre + four post')
    return blocks


def load_context(qid: str):
    if qid not in QIDS:
        raise ValueError('Only seven archived exact prompt contexts are eligible for E2: '+', '.join(QIDS))
    raw = (PROMPTS/f'{qid}_A_PROMPT.txt').read_text(encoding='utf-8')
    match = ORIGINAL.search(raw)
    if not match:
        raise ValueError(f'Cannot parse archived {qid} prompt')
    question = match.group(1).strip()
    blocks = source_blocks(match.group(2))
    # E2 stressor: same eight source passages, with post-cutoff versions first.
    # Both strict and weak arms see this IDENTICAL E2 context/order.
    reordered = sorted(blocks, key=lambda b: b['date']<=CUTOFF)
    return {
        'question': question, 'evidence': '\n\n'.join(b['text'] for b in reordered),
        'source_dates': {b['id']: b['date'].isoformat() for b in reordered},
        'original_prompt_sha256': sha(raw),
        'e2_order': [b['id'] for b in reordered],
        'e2_evidence_sha256': sha('\n\n'.join(b['text'] for b in reordered)),
    }


SCHEMA = '''Return ONLY a JSON object of the form:
{"claims":[{"id":"C01","text":"One atomic factual sentence.","citations":["exact_SOURCE_ID"]}],"abstention":false}
Use at most 8 independently checkable factual claim records, IDs C01,C02,... consecutively.
Cite only exact SOURCE IDs in the passages provided. If no claims can be supported, return {"claims":[],"abstention":true}.
Do not browse or retrieve external evidence.'''


def prompt_a(mode, context):
    if mode == 'strict':
        instruction = ('Write a historical scientific synthesis as of 2025-06-30 23:59:59 UTC. '
                       'NEVER cite a SOURCE version dated after this cutoff. '
                       'Judge dates at the supplied version level, not by paper title. '
                       'Support each factual claim with admissible cited evidence.')
    elif mode == 'weak':
        instruction = ('Write a scientific synthesis using the supplied passages. '
                       'Support each factual claim with cited evidence. '
                       'No historical source-date restriction is specified by this generator instruction; '
                       'the question itself is unchanged.')
    else:
        raise ValueError(mode)
    return (f'{instruction}\n\n{SCHEMA}\n\nQUESTION:\n{context["question"]}'
            f'\n\nEVIDENCE:\n{context["evidence"]}')


def prompt_b(context, baseline_a, arm):
    return ('''You are a deletion-only claim verifier for a scientific historical synthesis.
Historical cutoff: 2025-06-30 23:59:59 UTC. Use only the question, eight passages,
and the exact saved baseline-A output. Check each claim's entire factual text
against its cited source passages AND the cited source-version DATE. A citation to
post-cutoff material is temporally inadmissible even if the text is factually plausible.
For EACH original claim choose exactly one code:
KEEP, DROP_POST_CUTOFF, DROP_UNSUPPORTED, DROP_UNCERTAIN.
Do not add, alter, merge, reorder, or rewrite claims.
Output ONLY a JSON object with the single key "decisions":
{"decisions":{"C01":"KEEP"}}
Provide exactly one decision for EVERY original claim ID. Do not output
"retain_claim_ids": the scoring script deterministically derives retained
IDs from decisions marked KEEP. Decisions are authoritative.
No external facts or retrieval.\n\nQUESTION:\n'''
            +context['question']+'\n\nEVIDENCE:\n'+context['evidence']
            +'\n\nIMMUTABLE BASELINE '+arm.upper()+' JSON:\n'+json.dumps(baseline_a,ensure_ascii=False,indent=2))


def schema_a(ctx):
    """Ollama native format schema. Structural requirements are not factual validation."""
    source_ids = list(ctx['source_dates'])
    return {
        'type':'object', 'additionalProperties':False,
        'required':['claims','abstention'],
        'properties':{
            'claims':{'type':'array','maxItems':8,'items':{
                'type':'object','additionalProperties':False,
                'required':['id','text','citations'],
                'properties':{
                    'id':{'type':'string','pattern':'^C[0-9]{2}$'},
                    'text':{'type':'string','minLength':1},
                    'citations':{'type':'array','minItems':1, 'items':{'type':'string','enum':source_ids}}
                }}},
            'abstention':{'type':'boolean'}
        }
    }


def schema_b(baseline_a):
    """One decision for each generated claim; no redundant retained list."""
    ids=[c['id'] for c in baseline_a['claims']]
    return {
        'type':'object', 'additionalProperties':False,
        'required':['decisions'],
        'properties':{'decisions':{'type':'object','additionalProperties':False,
            'required':ids,
            'properties':{cid:{'type':'string','enum':sorted(DECISIONS)} for cid in ids}
        }}
    }


def parse_json_text(raw: str):
    """Allow code fences but preserve raw content in the saved API record."""
    value = raw.strip()
    if value.startswith('```'):
        lines = value.splitlines()
        if len(lines)>=3 and lines[-1].strip() == '```':
            value = '\n'.join(lines[1:-1])
    return json.loads(value)


def validate_a(a, ctx):
    if not isinstance(a,dict) or set(a) != {'claims','abstention'}:
        raise ValueError('A has invalid top-level schema')
    claims=a['claims']
    if not isinstance(claims,list) or len(claims)>8:
        raise ValueError('A claims invalid')
    if type(a['abstention']) is not bool or a['abstention'] != (len(claims)==0):
        raise ValueError('A abstention and claims disagree')
    for n,c in enumerate(claims,1):
        if not isinstance(c,dict) or set(c) != {'id','text','citations'}:
            raise ValueError('A claim schema invalid')
        if c['id'] != f'C{n:02d}' or not isinstance(c['text'],str) or not c['text'].strip():
            raise ValueError('A ID or text invalid')
        if not isinstance(c['citations'],list) or not c['citations'] or not all(
            isinstance(s,str) and s in ctx['source_dates'] for s in c['citations']):
            raise ValueError('A claim has missing or unknown citations')
    return a


def validate_b(b, baseline_a):
    if not isinstance(b,dict) or set(b) != {'retain_claim_ids','decisions'}:
        raise ValueError('B has invalid schema')
    ids=[c['id'] for c in baseline_a['claims']]
    if not isinstance(b['decisions'],dict) or set(b['decisions'])!=set(ids):
        raise ValueError('B must decide every original claim')
    if not all(x in DECISIONS for x in b['decisions'].values()):
        raise ValueError('Unknown B decision code')
    if b['retain_claim_ids'] != [x for x in ids if b['decisions'][x]=='KEEP']:
        raise ValueError('B has invalid retained sequence')
    return b


def canonicalize_b_v3(raw_b, baseline_a):
    """Turn model *decisions* into canonical B; NEVER change a decision label.

    A redundant retain_claim_ids field, if emitted despite the v3 prompt,
    is preserved verbatim in *_parsed_raw.json and reported in the audit.
    It is non-authoritative under the prospective E2v3 protocol.
    """
    if not isinstance(raw_b, dict) or set(raw_b) not in (
            {'decisions'}, {'decisions', 'retain_claim_ids'}):
        raise ValueError('B v3 requires decisions; optional redundant retain_claim_ids only')
    ids = [c['id'] for c in baseline_a['claims']]
    decisions = raw_b['decisions']
    if not isinstance(decisions, dict) or set(decisions) != set(ids):
        raise ValueError('B v3 must decide every original claim exactly once')
    if not all(type(code) is str and code in DECISIONS for code in decisions.values()):
        raise ValueError('B v3 has an unknown decision code')
    canonical = {'retain_claim_ids': [cid for cid in ids if decisions[cid]=='KEEP'],
                 'decisions': {cid:decisions[cid] for cid in ids}}
    audit = {'protocol_version':PROTOCOL_VERSION,
             'decision_labels_modified':False,
             'retained_ids_derivation':'original_order_claim_ids_with_decision_KEEP',
             'raw_retain_ids_field_present':'retain_claim_ids' in raw_b,
             'raw_retain_ids_field':raw_b.get('retain_claim_ids'),
             'derived_retain_ids':canonical['retain_claim_ids'],
             'redundant_field_conflict':False,
             'warnings':[]}
    if 'retain_claim_ids' in raw_b and raw_b['retain_claim_ids'] != canonical['retain_claim_ids']:
        audit['redundant_field_conflict']=True
        audit['warnings'].append('MODEL_EMITTED_CONTRADICTORY_REDUNDANT_RETAINED_IDS')
    return validate_b(canonical, baseline_a), audit


def rule_date_only(a, ctx):
    """Strict rule baseline: rejects ANY POST or UNKNOWN citation; no semantics assessed."""
    dates={sid:dt.datetime.fromisoformat(value) for sid,value in ctx['source_dates'].items()}
    choices={}
    for c in a['claims']:
        flags=[sid not in dates or dates[sid]>CUTOFF for sid in c['citations']]
        choices[c['id']] = 'DROP_POST_CUTOFF' if any(flags) else 'KEEP'
    return {'retain_claim_ids':[k for k,v in choices.items() if v=='KEEP'], 'decisions':choices}


def request_chat(url, model, prompt, temp, seed, timeout, api_key, output_schema=None):
    """Use Ollama's native structured-output API; keep hosted API unchanged.

    All raw provider responses are archived by the caller before validation.
    """
    is_native = url.rstrip('/').endswith('/api/chat')
    messages=[
        {'role':'system','content':'You are executing a controlled scientific synthesis experiment. Follow the user formatting instructions exactly.'},
        {'role':'user','content':prompt}
    ]
    if is_native:
        payload={'model':model,'messages':messages, 'stream':False,
                 'format':output_schema if output_schema is not None else 'json',
                 'options':{'temperature':temp}}
        if seed is not None:
            payload['options']['seed']=seed
    else:
        payload={'model':model,'messages':messages,'temperature':temp,'stream':False}
        if seed is not None:
            payload['seed']=seed  # backend support is model/provider-dependent
    req=urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                               headers={'Content-Type':'application/json'},method='POST')
    if api_key:
        req.add_header('Authorization', 'Bearer '+api_key)
    start=now()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content=response.read().decode('utf-8')
            status=response.status
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'API HTTP {e.code}: '+e.read().decode('utf-8','replace')[:350]) from e
    finish=now()
    obj=json.loads(content)
    try:
        if is_native:
            message=obj['message']['content']
        else:
            message=obj['choices'][0]['message']['content']
        if not isinstance(message,str): raise TypeError('Non-string model content')
    except (KeyError,IndexError,TypeError) as e:
        raise RuntimeError('API response missing model message content') from e
    return {'requested_at_utc':start,'received_at_utc':finish,'status':status,
            'api_response':obj,'raw_text':message,
            'transport':'OLLAMA_NATIVE_SCHEMA' if is_native else 'OPENAI_COMPATIBLE_UNCONSTRAINED'}


def check_frozen_reference_config():
    """Fail closed if the original locked protocol is not compatible with this extension."""
    cfg = json.loads((ROOT / '05_Protocol/config_locked.json').read_text(encoding='utf8'))
    if cfg.get('cutoff_utc') != '2025-06-30T23:59:59Z':
        raise ValueError('Original cutoff changed; abort E2')
    if cfg.get('context',{}).get('per_question') != 8 or cfg['context'].get('pre_count') != 4 or cfg['context'].get('post_count') != 4:
        raise ValueError('Original context format changed; abort E2')
    if not set(QIDS).issubset(set(cfg.get('question_ids',{}).get('HELD_OUT', []))):
        raise ValueError('E2 source questions not present in the original frozen protocol')
    return cfg


def experiment(args):
    check_frozen_reference_config()
    qids=QIDS if args.all else (args.qid,)
    if not args.dry_run and not args.model:
        raise ValueError('--model is required for a real run')
    urls={
        'ollama':'http://127.0.0.1:11434/api/chat',
        'openai':'https://api.openai.com/v1/chat/completions',
        'gemini':'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'
    }
    url=args.url or urls[args.backend]
    envname=args.api_key_env or ('GEMINI_API_KEY' if args.backend=='gemini' else 'OPENAI_API_KEY')
    key=os.environ.get(envname, '') if args.backend!='ollama' else ''
    if args.backend!='ollama' and not args.dry_run and not key:
        raise ValueError('No key in the selected environment variable (never store API keys in project files)')
    for qid in qids:
        context=load_context(qid)
        if args.dry_run:
            print(qid,'source order:',', '.join(context['e2_order']))
            print('  exact original prompt sha256:',context['original_prompt_sha256'])
            print('  E2 evidence sha256:',context['e2_evidence_sha256'])
            print('  strict/weak prompt sha256:',sha(prompt_a('strict',context)),sha(prompt_a('weak',context)))
            continue
        for r in range(1,args.repeats+1):
            # unique immutable run path; separate fresh HTTP messages for every arm
            run_id=f'{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S%fZ}_{qid}_r{r}_{uuid.uuid4().hex[:8]}'
            folder=OUTDIR/run_id
            folder.mkdir(parents=True,exist_ok=False)
            seed=args.seed+r-1 if args.seed is not None else None
            metadata={'study':'E2_POST_FIRST_EXTENSION_NOT_E1_RERUN', 'protocol_version':PROTOCOL_VERSION,
                      'created_utc':now(),
                      'qid':qid,'repeat':r,'seed_requested':seed,
                      'model_requested':args.model,'backend':args.backend,'url':url,
                      'temperature_requested':args.temperature,'model_seed_honoring_UNVERIFIED':True,
                      'context':context, 'status':'STARTED'}
            dump(folder/'metadata.json',metadata)
            try:
                for arm in ('strict','weak'):
                    p=prompt_a(arm,context)
                    output_schema=schema_a(context)
                    dump(folder/f'{arm}_request.json',{'prompt':p,'prompt_sha256':sha(p),
                         'output_schema':output_schema,'output_schema_sha256':sha(json.dumps(output_schema,sort_keys=True)),
                         'temperature':args.temperature, 'seed':seed, 'endpoint':url})
                    resp=request_chat(url,args.model,p,args.temperature,seed,args.timeout,key,output_schema)
                    dump(folder/f'{arm}_raw_api.json',resp)
                    parsed=validate_a(parse_json_text(resp['raw_text']),context)
                    dump(folder/f'{arm}_parsed.json',parsed)
                weak=json.loads((folder/'weak_parsed.json').read_text(encoding='utf-8'))
                strict=json.loads((folder/'strict_parsed.json').read_text(encoding='utf-8'))
                ruled=rule_date_only(weak,context)
                dump(folder/'date_rule_decisions.json',ruled)
                # E2v4: Native Ollama JSON-schema output; B still returns decisions only.
                # Raw responses and any contradictory redundant field remain archived.
                warnings=[]
                for baseline_arm, baseline_json in (('weak',weak),('strict',strict)):
                    p=prompt_b(context,baseline_json,baseline_arm)
                    prefix=f'{baseline_arm}_llm_verifier'
                    output_schema=schema_b(baseline_json)
                    dump(folder/f'{prefix}_request.json',{'prompt':p,'prompt_sha256':sha(p),
                         'output_schema':output_schema,'output_schema_sha256':sha(json.dumps(output_schema,sort_keys=True)),
                         'temperature':args.temperature,'seed':seed,'endpoint':url})
                    resp=request_chat(url,args.model,p,args.temperature,seed,args.timeout,key,output_schema)
                    dump(folder/f'{prefix}_raw_api.json',resp)
                    raw_b=parse_json_text(resp['raw_text'])
                    dump(folder/f'{prefix}_parsed_raw.json',raw_b)
                    b,audit=canonicalize_b_v3(raw_b,baseline_json)
                    dump(folder/f'{prefix}_canonicalization_audit.json',audit)
                    dump(folder/f'{prefix}_decisions.json',b)
                    warnings.extend([baseline_arm+': '+w for w in audit['warnings']])
                dump(folder/'COMPLETED.json', {'completed_utc':now(),
                                               'status':'COMPLETE_WITH_FORMAT_WARNINGS' if warnings else 'COMPLETE_VALID_FORMAT',
                                               'protocol_version':PROTOCOL_VERSION,
                                               'format_warnings':warnings,
                                               'claim_support':'NOT_INDEPENDENTLY_ANNOTATED',
                                               'gold_coverage':'NOT_AVAILABLE'})
                print('COMPLETE',qid,'repeat',r,folder)
            except Exception as e:
                dump(folder/'FAILED.json',{'time_utc':now(),'error':repr(e),'status':'INCOMPLETE_DO_NOT_SCORE_AS_ZERO'})
                print('FAILED',qid,'repeat',r,':',e,file=sys.stderr)
                if not args.continue_on_error: raise
    if args.dry_run:
        print('DRY RUN ONLY — NO MODEL CALLED; NO EXPERIMENTAL RESULTS CREATED')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--qid', choices=QIDS)
    g.add_argument('--all',action='store_true')
    p.add_argument('--backend',choices=('ollama','openai','gemini'),default='gemini')
    p.add_argument('--url',help='API endpoint (for Ollama use native /api/chat for JSON schema enforcement)')
    p.add_argument('--model',help='Remote API or installed local model ID (e.g. atr-qwen3b)')
    p.add_argument('--api-key-env',default=None,help='Environment variable containing the provider API key; default GEMINI_API_KEY for Gemini, OPENAI_API_KEY otherwise')
    p.add_argument('--temperature',type=float,default=0.0)
    p.add_argument('--seed',type=int,help='Optional base seed (backend may ignore/reject)')
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--timeout',type=int,default=240)
    p.add_argument('--continue-on-error',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.repeats<1: p.error('--repeats must be >= 1')
    if not (0<=a.temperature<=2):p.error('--temperature must be between 0 and 2')
    try: experiment(a)
    except Exception as e: p.exit(1, f'ERROR: {e}\n')

if __name__=='__main__': main()
