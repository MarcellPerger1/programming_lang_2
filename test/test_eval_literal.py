import ast
import sys

from parser.astgen.eval_literal import eval_string, AstStringParseError
from parser.common import StrRegion
from test.common import CommonTestCase


class TestEvalStringConsistency(CommonTestCase):
    # noinspection PyMethodMayBeStatic
    def _get_result_from_full(self, s: str):
        return eval_string(f"{s}", StrRegion(2, 2 + len(s)), f"a={s};")

    def test_basic(self):
        self.assertEqual('a', self._get_result_from_full("'a'"))

    def test_error(self):
        with self.assertRaises(AstStringParseError) as ctx:
            self._get_result_from_full(r'"\x7"')
        self.assertContains(ctx.exception.msg, "expected 2")
        self.assertEqual(3, ctx.exception.region.start)
        self.assertEqual(6, ctx.exception.region.end)

    def test_py_consistency(self):
        base = r'a\\\a\b\v\f\0\n\rq\t' '\\"' "\\'"
        for x in range(0, 256, 3):
            for u in range(0, 65000, 1217):
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
