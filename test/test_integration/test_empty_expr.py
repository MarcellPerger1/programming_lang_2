import unittest
from typing import Iterable, Container, Any

from test.snapshottest import SnapshotTestCase
from parser.tokenizer import Tokenizer
from parser.cst.treegen import CstGen
from parser.error import CstParseError


class MyTestCase(SnapshotTestCase):
    def assertContains(self, container: Iterable | Container, member: Any, msg=None):
        self.assertIn(member, container, msg)

    def test_error_empty_assign_source(self):
        t = Tokenizer('let a= ;').tokenize()
        with self.assertRaises(CstParseError) as err:
            CstGen(t).parse()
        self.assertContains(str(err.exception), "expr")

    def test_error_empty_condition(self):
        t = Tokenizer('if {x();}').tokenize()
        with self.assertRaises(CstParseError) as err:
            CstGen(t).parse()
        self.assertContains(str(err.exception), "expr")

    def test_empty_expr_issue_04(self):
        t = Tokenizer('let a=9;;').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node)


if __name__ == '__main__':
    unittest.main()
