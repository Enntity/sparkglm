# SPDX-License-Identifier: AGPL-3.0-only
"""Supplemental source-review findings, explicitly written after model responses."""
import unittest
from test_candidate import candidate,oracle

@unittest.skipUnless(candidate,'Set reviewed candidate path')
class ReviewFindingTests(unittest.TestCase):
    def test_disjoint_duplicate_window_ids(self):
        with self.assertRaises(ValueError):candidate([], [dict(id='same',start=0,end=1),dict(id='same',start=2,end=3)])
    def test_integer_timestamp_precision(self):
        base=10**18
        e=[dict(name='a',start=base+1,end=base+2)]
        w=[dict(id='w',start=base,end=base+3)]
        self.assertEqual(candidate(e,w),oracle(e,w))
    def test_large_integer_computed_overflow(self):
        value=10**308
        with self.assertRaises(ValueError):candidate([],[dict(id='w',start=-value,end=value)])

if __name__=='__main__':unittest.main()
