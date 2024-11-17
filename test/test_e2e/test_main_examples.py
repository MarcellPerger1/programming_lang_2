import unittest

from parser.astgen.astgen import AstGen
from parser.cst.cstgen import CstGen
from parser.lexer import Tokenizer
from test.common import CommonTestCase
from util import readfile


class TestMain(CommonTestCase):
    def setUp(self):
        self.setProperCwd()
        super().setUp()

    def _test_main_example_n(self, n: int, do_ast: bool = True):
        src = readfile(f'./main_example_{n}.st')
        tk = Tokenizer(src).tokenize()
        self.assertMatchesSnapshot(tk.tokens, 'tokens')
        t = CstGen(tk)
        self.assertMatchesSnapshot(t.parse(), 'cst')
        if do_ast:
            self.assertMatchesSnapshot(AstGen(t).parse(), 'ast')

    def test_example_0(self):
        self._test_main_example_n(0, do_ast=False)

    def test_example_1(self):
        self._test_main_example_n(1)


if __name__ == '__main__':
    unittest.main()
