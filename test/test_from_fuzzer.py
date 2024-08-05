import unittest

from parser.lexer.tokenizer import Tokenizer
from parser.cst.treegen import TreeGen
from parser.error import BaseParseError


class FuzzerTestCase(unittest.TestCase):
    """Tests for crashes caught by pythonfuzz"""

    def assertNotInternalError(self, src: str):
        try:
            TreeGen(Tokenizer(src)).parse()
        except BaseParseError:
            pass

    def test_3610f59833246958fff7d5cbc5b23f8c99496c3c8fda3f5606f5b198713cbb95(self):
        self.assertNotInternalError('. ')
        self.assertNotInternalError('.')


if __name__ == '__main__':
    unittest.main()
