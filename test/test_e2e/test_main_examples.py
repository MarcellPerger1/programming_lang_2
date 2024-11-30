import unittest

from parser.astgen.astgen import AstGen
from parser.cst.cstgen import CstGen
from parser.lexer import Tokenizer
from parser.typecheck.typecheck import NameResolver
from test.common import CommonTestCase
from util import readfile


class TestMain(CommonTestCase):
    def setUp(self):
        self.setProperCwd()
        super().setUp()

    def _test_main_example_n(self, n: int, do_ast=True, do_name_resolve=True):
        src = readfile(f'./main_example_{n}.st')
        tk = Tokenizer(src).tokenize()
        self.assertMatchesSnapshot(tk.tokens, 'tokens')
        t = CstGen(tk)
        self.assertMatchesSnapshot(t.parse(), 'cst')
        if not do_ast:
            return
        a = AstGen(t)
        self.assertMatchesSnapshot(a.parse(), 'ast')
        if not do_name_resolve:
            return
        nr = NameResolver(a)
        self.assertMatchesSnapshot(nr.run(), 'name_resolve')

    def test_example_0(self):
        self._test_main_example_n(0, do_ast=False)

    def test_example_1(self):
        self._test_main_example_n(1, do_name_resolve=False)

    def test_example_2(self):
        self._test_main_example_n(2)


if __name__ == '__main__':
    unittest.main()
