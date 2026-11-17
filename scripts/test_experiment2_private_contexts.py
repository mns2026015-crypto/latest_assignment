"""Synthetic software tests only. They are not scientific results."""
import sys
import unittest
import json
import os
import tempfile
import types
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_experiment as runner
from run_experiment import QIDS, load_context, prompt_a, validate_a, validate_b, rule_date_only, parse_json_text, canonicalize_b_v3, schema_a, schema_b, request_chat
from score_experiment2 import metrics

class Experiment2Tests(unittest.TestCase):
    def test_seven_contexts_have_post_first(self):
        for qid in QIDS:
            context=load_context(qid)
            dates=list(context['source_dates'].values())
            self.assertEqual(len(dates),8)
            self.assertTrue(all(v[:10]>'2025-06-30' for v in dates[:4]))
            self.assertTrue(all(v[:10]<='2025-06-30' for v in dates[4:]))
            self.assertNotEqual(prompt_a('strict',context),prompt_a('weak',context))
    def test_rule_and_delete_guards(self):
        c=load_context('H01');ids=c['e2_order']
        a={'claims':[
            {'id':'C01','text':'Provisional source claim.','citations':[ids[0]]},
            {'id':'C02','text':'Valid old version claim.','citations':[ids[-1]]},
            {'id':'C03','text':'Mixed references.','citations':[ids[0],ids[-1]]}],
           'abstention':False}
        validate_a(a,c)
        decision=rule_date_only(a,c)
        self.assertEqual(decision['retain_claim_ids'],['C02'])
        validate_b(decision,a)
        m=metrics(a['claims'],c['source_dates'])
        self.assertEqual((m['pre_only_cited'],m['post_only_cited'],m['mixed_date_cited']),(1,1,1))
        decision['retain_claim_ids'].append('C03')
        with self.assertRaises(ValueError):validate_b(decision,a)
    def test_json_and_abstention(self):
        self.assertEqual(parse_json_text('```json\n{"ok":true}\n```'),{'ok':True})
        ctx=load_context('H02')
        validate_a({'claims':[],'abstention':True},ctx)
        with self.assertRaises(ValueError):validate_a({'claims':[],'abstention':False},ctx)

    def test_v3_decisions_only_and_conflicting_extra_field(self):
        a={'claims':[{'id':'C01','text':'A','citations':['S01']},
                     {'id':'C02','text':'B','citations':['S02']}], 'abstention':False}
        raw={'decisions':{'C01':'DROP_UNCERTAIN','C02':'KEEP'}}
        canonical,audit=canonicalize_b_v3(raw,a)
        self.assertEqual(canonical['retain_claim_ids'],['C02'])
        self.assertEqual(audit['warnings'],[])
        self.assertFalse(audit['decision_labels_modified'])
        conflicted={'decisions':{'C01':'DROP_UNCERTAIN','C02':'KEEP'},
                    'retain_claim_ids':['C01','C02']}
        canonical,audit=canonicalize_b_v3(conflicted,a)
        self.assertEqual(canonical['retain_claim_ids'],['C02'])
        self.assertTrue(audit['redundant_field_conflict'])
        self.assertEqual(audit['raw_retain_ids_field'],['C01','C02'])
        with self.assertRaises(ValueError):
            canonicalize_b_v3({'decisions':{'C01':'KEEP'}},a)

    def test_a_requires_citations_in_ollama_schema(self):
        ctx=load_context('H02')
        schema=schema_a(ctx)
        claim=schema['properties']['claims']['items']
        self.assertEqual(set(claim['required']),{'id','text','citations'})
        self.assertGreaterEqual(claim['properties']['citations']['minItems'],1)
        self.assertEqual(set(claim['properties']['citations']['items']['enum']),set(ctx['source_dates']))
        with self.assertRaisesRegex(ValueError,'A claim schema invalid'):
            validate_a({'claims':[{'id':'C01','text':'NO CITATION'}],'abstention':False},ctx)

    def test_b_schema_requires_exact_claim_ids(self):
        a={'claims':[{'id':'C01','text':'A','citations':['S01']},
                     {'id':'C02','text':'B','citations':['S02']}], 'abstention':False}
        schema=schema_b(a)
        decision=schema['properties']['decisions']
        self.assertEqual(decision['required'],['C01','C02'])
        self.assertFalse(decision['additionalProperties'])

    def test_ollama_native_transport_uses_format_schema(self):
        ctx=load_context('H02')
        output_schema=schema_a(ctx)
        requests=[]
        class FakeHTTPResponse:
            status=200
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def read(self):
                return json.dumps({'message':{'content':json.dumps({'claims':[],'abstention':True})}}).encode()
        def fake_urlopen(request,timeout=None):
            requests.append(json.loads(request.data))
            self.assertEqual(request.full_url,'http://127.0.0.1:11434/api/chat')
            return FakeHTTPResponse()
        with patch.object(runner.urllib.request,'urlopen',fake_urlopen):
            res=request_chat('http://127.0.0.1:11434/api/chat','atr-qwen3b',
                         'sample prompt',0.0,None,5,'',output_schema)
        self.assertEqual(res['transport'],'OLLAMA_NATIVE_SCHEMA')
        self.assertEqual(requests[0]['format'],output_schema)
        self.assertEqual(requests[0]['options']['temperature'],0.0)
        self.assertEqual(json.loads(res['raw_text'])['claims'],[])

    def test_full_run_offline_mock_hosted_provider(self):
        """No external HTTP; verifies four independent calls, raw retention and completion gate."""
        calls=[]
        context=load_context('H01')
        def fake_api(url, model, prompt, temp, seed, timeout, key, output_schema=None):
            self.assertIsInstance(output_schema, dict)
            calls.append(prompt)
            if 'IMMUTABLE BASELINE' in prompt:
                response={'decisions':{'C01':'KEEP'}}
            else:
                response={'claims':[{'id':'C01','text':'Synthetic fixture only.',
                                     'citations':[context['e2_order'][0]]}], 'abstention':False}
            return {'requested_at_utc':'TEST','received_at_utc':'TEST',
                    'status':200,'api_response':{'model':model,'choices':[]},
                    'raw_text':json.dumps(response)}
        with tempfile.TemporaryDirectory() as tmp:
            args=types.SimpleNamespace(all=False,qid='H01',dry_run=False,model='fixture-model',
                backend='gemini',url=None,api_key_env=None,repeats=1,
                seed=None,temperature=0.0,timeout=4,continue_on_error=False)
            with patch.object(runner,'OUTDIR',Path(tmp)), patch.object(runner,'request_chat',fake_api), \
                patch.dict(os.environ,{'GEMINI_API_KEY':'MOCK_NOT_REAL_KEY'}):
                runner.experiment(args)
            paths=list(Path(tmp).glob('*/COMPLETED.json'))
            self.assertEqual(len(paths),1)
            self.assertEqual(len(calls),4)
            folder=paths[0].parent
            for file in ('strict_parsed.json','weak_parsed.json','strict_llm_verifier_decisions.json',
                         'weak_llm_verifier_decisions.json','date_rule_decisions.json',
                         'strict_llm_verifier_canonicalization_audit.json',
                         'weak_llm_verifier_canonicalization_audit.json'):
                self.assertTrue((folder/file).is_file(),file)

if __name__ == '__main__':unittest.main()
