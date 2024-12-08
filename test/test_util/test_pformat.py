import enum

from test.common import CommonTestCase
from util.pformat import pformat


class MyEnum(enum.Enum):
    FOO = 'foo'
    BAR = 'bar'


def dedent(s: str, keep_empty_ends=False):
    res = s.splitlines()
    if res[0].lstrip() == '':
        del res[0]
    if res[-1].lstrip() == '':
        del res[-1]
    by = min(next(i for (i, c) in enumerate(ln) if not c.isspace())
             for ln in res if ln and not ln.isspace())
    return dedent_by(s, by, keep_empty_ends)


def dedent_by(s: str, by: int, keep_empty_ends=False):
    res = s.splitlines()
    if not keep_empty_ends:
        if res[0].lstrip() == '':
            del res[0]
        if res[-1].lstrip() == '':
            del res[-1]
    for i, ln in enumerate(res):
        start = next((j for j, c in enumerate(ln) if not c.isspace() or j >= by), len(ln))
        res[i] = ln[start:]
    return '\n'.join(res)


class TestPFormat(CommonTestCase):
    def test_fmt_enum(self):
        self.assertEqual('MyEnum.FOO', pformat(MyEnum.FOO))

    def test_fmt_short(self):
        self.assertEqual('[MyEnum.FOO, 6]', pformat([MyEnum.FOO, 6]))

    def test_fmt_tuple(self):
        self.assertEqual('(MyEnum.FOO,)', pformat((MyEnum.FOO,)))
        self.assertEqual('(MyEnum.FOO, 6)', pformat((MyEnum.FOO, 6)))
        self.assertEqual(dedent('''
        (
          MyEnum.FOO,
          (4, 5)
        )'''), pformat((MyEnum.FOO, (4, 5))))
        self.assertEqual(dedent('''
        (
          (
            2,
            3,
            (4,)
          ),
        )'''), pformat(((2, 3, (4,)),)))
        self.assertEqual(dedent('''
        (
            MyEnum.FOO,
            (4, 5)
        )'''), pformat((MyEnum.FOO, (4, 5)), indent=4))
        self.assertEqual(dedent('''
        (
            (
                2,
                3,
                (4,)
            ),
        )'''), pformat(((2, 3, (4,)),), indent=4))

    def test_set(self):
        self.assertEqual('set()', pformat(set()))
        self.assertEqual('{-8, -2, 1, 2, 5}', pformat({2, 5, -8, 1, -2}))
