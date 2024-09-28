import unittest

from parser.util import readfile
from parser.cst.treegen import TreeGen
from parser.lexer import Tokenizer
from test.snapshottest import SnapshotTestCase
from test.utils import TestCaseUtils


class TestMain(TestCaseUtils, SnapshotTestCase):
    def setUp(self):
        self.setProperCwd()
        super().setUp()

    def _test_main_example_n(self, n: int):
        src = readfile(f'./main_example_{n}.st')
        tk = Tokenizer(src).tokenize()
        self.assertMatchesSnapshot(tk.tokens, 'tokens')
        t = TreeGen(tk)
        self.assertMatchesSnapshot(t.parse(), 'cst')

    def test_example_0(self):
        self._test_main_example_n(0)

    def test_example_1(self):
        self._test_main_example_n(1)


if __name__ == '__main__':
    unittest.main()
