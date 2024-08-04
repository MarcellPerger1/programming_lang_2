import unittest

from parser.lexer.tokenizer import Tokenizer
from parser.cst.treegen import TreeGen
from parser.cst.tree_print import tprint

from test.snapshottest import SnapshotTestCase


class TreeGenTest(SnapshotTestCase):
    maxDiff = None

    def test_item_chain(self):
        tk = Tokenizer('a[7].b.0.fn["c" .. 2] = fn(9).k[7 + r](3,);')
        t = TreeGen(tk)
        t.parse()
        tprint(t.result)
        self.assertMatchesSnapshot(t.result)

    def test_fn_call_in_lvalue(self):
        tk = Tokenizer('a(7).b.0.fn()["c" .. 2] = fn(9).k[7 + r](3,);')
        t = TreeGen(tk)
        t.parse()
        tprint(t.result)
        self.assertMatchesSnapshot(t.result)

    def test_aug_assign(self):
        tk = Tokenizer('a[1] += a.2;')
        t = TreeGen(tk)
        t.parse()
        tprint(t.result)
        self.assertMatchesSnapshot(t.result)


if __name__ == '__main__':
    unittest.main()
