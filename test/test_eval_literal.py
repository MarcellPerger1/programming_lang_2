import ast
import sys

from parser.astgen.eval_literal import eval_string, eval_number
from parser.astgen.errors import AstStringParseError
from parser.common import StrRegion
from test.common import CommonTestCase


class TestEvalNumber(CommonTestCase):
    def test_float(self):
        v = eval_number('.3e-7')
        self.assertEqual(0.3e-7, v)
        self.assertIsInstance(v, float)
        v = eval_number('2.0')
        self.assertEqual(2.0, v)
        self.assertIsInstance(v, float)

    def test_int(self):
        v = eval_number('6')
        self.assertEqual(6, v)
        self.assertIsInstance(v, int)
        v = eval_number('-122')
        self.assertEqual(-122, v)
        self.assertIsInstance(v, int)


class TestEvalStringConsistency(CommonTestCase):
    # noinspection PyMethodMayBeStatic
    def _get_result_from_full(self, s: str):
        return eval_string(f"{s}", StrRegion(2, 2 + len(s)), f"a={s};")

    def test_basic(self):
        self.assertEqual('a', self._get_result_from_full("'a'"))

    def _assert_err_and_region(self, src: str, region: StrRegion):
        with self.assertRaises(AstStringParseError) as ctx:
            self._get_result_from_full(src)
        self.assertEqual(region, ctx.exception.region)
        return ctx.exception.msg

    def test_error(self):
        msg = self._assert_err_and_region(r'"\x7"', StrRegion(3, 6))
        self.assertContains(msg, "expected 2")
        self.assertContains(msg.lower(), "unterminated")

        msg = self._assert_err_and_region(r'"\u4fdq"', StrRegion(3, 9))
        self.assertContains(msg, "expected 4")

        msg = self._assert_err_and_region(r'"\q"', StrRegion(3, 5))
        self.assertContains(msg, "Unknown string escape \\q")

        msg = self._assert_err_and_region(r'"\N"', StrRegion(3, 5))
        self.assertContains(msg, "\\N")
        self.assertContains(msg.lower(), "expected '{'")

        msg = self._assert_err_and_region(r'"\N7"', StrRegion(3, 5))
        self.assertContains(msg, "\\N")
        self.assertContains(msg.lower(), "expected '{'")

        msg = self._assert_err_and_region(r'"\N{}"', StrRegion(3, 7))
        self.assertContains(msg.lower(), "empty character name")

        msg = self._assert_err_and_region(r'"\N{EM DASH"', StrRegion(3, 3 + 10))
        self.assertContains(msg.lower(), "terminated by a '}'")

        msg = self._assert_err_and_region(
            r'"\N{i am not a unicode character and if i existed these '
            r'tests would break}"', StrRegion(3, 3 + 73))
        self.assertContains(msg.lower(), "unknown unicode character name")

    def test_bad_hex_escape_2(self):
        msg = self._assert_err_and_region(r'"\x-9"', StrRegion(3, 7))
        self.assertContains(msg, "expected 2")

        msg = self._assert_err_and_region(r'"\u 4fe1"', StrRegion(3, 9))
        self.assertContains(msg, "expected 4")

    def test_py_consistency(self):
        base = r'a\\\a\b\v\f\0\n\rq\t' '\\"' "\\'"
        for x in (*range(0, 34, 3), *range(34, 256, 9), 255):
            for u in range(0, 65000, 1387):
                for big_u in range(0, 0x110000, 0xE4C1):
                    v2 = (base + f'\\x{x:0>2x}1\\u{u:0>4x}e\\U{big_u:0>8x}'
                                 '\\N{COPYRIGHT SIGN}\\N{EM DASH}')
                    for q in ('"', "'"):
                        v3 = q + v2 + q
                        src = 'abc=' + v3 + ';\nprint(abc);'
                        r = StrRegion(4, 4 + len(v3))
                        self._check_consistent_once(v3, r, src, x, u, big_u, q)

    def _check_consistent_once(self, string_part, reg, src, x, u, big_u, q):
        my = eval_string(string_part, reg, src)
        py = ast.literal_eval(string_part)
        if my != py:
            print(
                f'Consistency check failed:\n'
                f'expected={py!r}\n'
                f'actual={my!r}\n'
                f'({x=}, {u=}, U={big_u}, {q=})\n'
                f'(src is "{src}")',
                file=sys.stderr)
            self.assertEqual(py, my, "eval_string consistency "
                                     "check failed (see above)")
