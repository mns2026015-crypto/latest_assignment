import unittest
from recompute_public_dates import summarize,EXPECTED
class AuditTest(unittest.TestCase):
    def test_expected_values(self):self.assertEqual(summarize(),EXPECTED)
    def test_arm_totals(self):
        r=summarize()
        self.assertEqual(r['A_plus_Qwen'][1],16)
        self.assertEqual(r['A0_plus_rule'][-1],0)
        self.assertEqual(r['A0_plus_Qwen'][-1],13)
if __name__=='__main__': unittest.main()
