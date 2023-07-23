import unittest

from parser.cst.treegen import CstGen
from parser.tokenizer import Tokenizer
from test.snapshottest import SnapshotTestCase


class MyTestCase(SnapshotTestCase):
    def test_issue_09__dot(self):
        t = Tokenizer('fn(call_arg).a;').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node, 'after_call')
        t = Tokenizer('(paren + x).b').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node, 'after_paren')
        t = Tokenizer('"a string".b;').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node, 'after_string')

    def test_issue_09__sqb(self):
        t = Tokenizer('fn(call_arg)[1];').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node, 'after_call')
        t = Tokenizer('(paren + x)[2]').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node, 'after_paren')
        t = Tokenizer('"a string"["key_" .. 3];').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node, 'after_string')


if __name__ == '__main__':
    unittest.main()
