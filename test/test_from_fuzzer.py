import asyncio
import unittest
from pathlib import Path

from parser.lexer.tokenizer import Tokenizer
from parser.cst.treegen import TreeGen
from parser.error import BaseParseError

from test.utils import timeout_decor, TestCaseUtils, timeout_decor_async


class BaseFuzzerTestCase(unittest.TestCase):
    def assertNotInternalError(self, src: str):
        try:
            TreeGen(Tokenizer(src)).parse()
        except BaseParseError:
            self.assertTrue(True)
        self.assertTrue(True)


# ~5x faster than sync version (3.6s vs 17.9s)
class FuzzerCorpusTestCases(unittest.IsolatedAsyncioTestCase, TestCaseUtils):
    def setUp(self):
        self.setProperCwd()

    @staticmethod
    def raiseInternalErrorOrNone(src: str):
        try:
            TreeGen(Tokenizer(src)).parse()
        except BaseParseError:
            return None
        except Exception as ex:
            raise ex
        return None

    # We need extra inner methods that MUST be static so that the unpicklable
    # TestCase object isn't passed to the other process as the self value
    @staticmethod
    @timeout_decor_async(5, debug=0, pool=True)
    def _inner_once(src: str):
        return FuzzerTimeoutTestCase.raiseInternalErrorOrNone(src)

    async def _test_once(self, p: Path):
        with self.subTest(corp=p.name):
            with open(p, encoding='cp1252') as f:
                src = f.read()
            # noinspection PyUnresolvedReferences
            await self._inner_once(src)  # Pycharm doesn't understand decorators

    async def test(self):
        await asyncio.gather(*[
            self._test_once(p) for p in Path('./pythonfuzz_corpus').iterdir()])


class FuzzerCrashTestCase(BaseFuzzerTestCase):
    """Tests for crashes caught by pythonfuzz"""

    def test_3610f59833246958fff7d5cbc5b23f8c99496c3c8fda3f5606f5b198713cbb95(self):
        self.assertNotInternalError('. ')
        self.assertNotInternalError('.')

    def test_62428fae4dba4c7722b8d7e1b8ad7bbe7a01ac1603e4421d42df7e0c0ad70f85(self):
        self.assertNotInternalError('!W>W>W9Jd')
        self.assertNotInternalError('!W>W>W9Jd\x1e')


class FuzzerTimeoutTestCase(BaseFuzzerTestCase):
    """Tests for timeouts caught by pythonfuzz"""

    @staticmethod
    def raiseInternalErrorOrNone(src: str):
        try:
            TreeGen(Tokenizer(src)).parse()
        except BaseParseError:
            return None
        except Exception as ex:
            raise ex
        return None

    # We need extra inner methods that MUST be static so that the unpicklable
    # TestCase object isn't passed to the other process as the self value
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
