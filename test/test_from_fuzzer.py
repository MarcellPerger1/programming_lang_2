import unittest

from parser.lexer.tokenizer import Tokenizer
from parser.cst.treegen import TreeGen
from parser.error import BaseParseError

from test.utils import timeout_decor


class FuzzerCrashTestCase(unittest.TestCase):
    """Tests for crashes caught by pythonfuzz"""

    def assertNotInternalError(self, src: str):
        try:
            TreeGen(Tokenizer(src)).parse()
        except BaseParseError:
            pass

    def test_3610f59833246958fff7d5cbc5b23f8c99496c3c8fda3f5606f5b198713cbb95(self):
        self.assertNotInternalError('. ')
        self.assertNotInternalError('.')

    def test_62428fae4dba4c7722b8d7e1b8ad7bbe7a01ac1603e4421d42df7e0c0ad70f85(self):
        self.assertNotInternalError('!W>W>W9Jd')
        self.assertNotInternalError('!W>W>W9Jd\x1e')


class FuzzerTimeoutTestCase(unittest.TestCase):
    def assertNotInternalError(self, src: str):
        try:
            TreeGen(Tokenizer(src)).parse()
        except BaseParseError:
            pass

    @staticmethod
    def raiseInternalErrorOrNone(src: str):
        try:
            TreeGen(Tokenizer(src)).parse()
        except BaseParseError:
            return None
        except Exception as ex:
            raise ex
        return None

    @staticmethod
    @timeout_decor(5, debug=0)
    def inner_a32460d584fd8a20d1e62007db570eaf41342f248e42c21a33780fd976e45290():
        return FuzzerTimeoutTestCase.raiseInternalErrorOrNone('s/*y')

    def test_a32460d584fd8a20d1e62007db570eaf41342f248e42c21a33780fd976e45290(self):
        self.inner_a32460d584fd8a20d1e62007db570eaf41342f248e42c21a33780fd976e45290()

    @staticmethod
    @timeout_decor(5, debug=0)
    def inner_ed988ae940f54542ec54fd3c402a009fe2fdb660bd558d76a3612781a5ef6aa2():
        TreeGen(Tokenizer('a;//')).parse()
        return FuzzerTimeoutTestCase.raiseInternalErrorOrNone('a<//')

    def test_ed988ae940f54542ec54fd3c402a009fe2fdb660bd558d76a3612781a5ef6aa2(self):
        self.inner_ed988ae940f54542ec54fd3c402a009fe2fdb660bd558d76a3612781a5ef6aa2()


if __name__ == '__main__':
    unittest.main()
