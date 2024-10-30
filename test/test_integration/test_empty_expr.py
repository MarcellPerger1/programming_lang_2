import unittest

from test.common import CommonTestCase
from parser.lexer import Tokenizer
from parser.cst.treegen import CstGen, LocatedCstError


class MyTestCase(CommonTestCase):

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
