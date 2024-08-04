import unittest
from typing import Iterable, Container, Any

from test.snapshottest import SnapshotTestCase
from parser.lexer import Tokenizer
from parser.cst.treegen import CstGen, LocatedCstError


class MyTestCase(SnapshotTestCase):
    def assertContains(self, container: Iterable | Container, member: Any, msg=None):
        self.assertIn(member, container, msg)

    # Typing this would be way too complex (a:LE | b:GE) & (b:LE | c: GE) if incl
    def assertBetweenIncl(self, lo, hi, value):
        self.assertGreaterEqual(value, lo)
        self.assertLessEqual(value, hi)

    def test_error_empty_assign_source(self):
        t = Tokenizer('let a= ;').tokenize()
        with self.assertRaises(LocatedCstError) as err:
            CstGen(t).parse()
        self.assertBetweenIncl(5, 7, err.exception.region.start)
        self.assertBetweenIncl(7, 8, err.exception.region.end)
        self.assertContains(str(err.exception), "semi")

    def test_error_empty_condition(self):
        t = Tokenizer('if {x();}').tokenize()
        with self.assertRaises(LocatedCstError) as err:
            CstGen(t).parse()
        self.assertBetweenIncl(0, 3, err.exception.region.start)
        self.assertBetweenIncl(2, 4, err.exception.region.end)
        self.assertContains(str(err.exception), "brace")

    def test_empty_expr_issue_04(self):
        t = Tokenizer('let a=9;;').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node)


if __name__ == '__main__':
    unittest.main()
