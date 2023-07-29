import unittest
from enum import IntFlag, FlagBoundary, IntEnum
from typing import Sequence, TypeVar, Union

from parser.str_region import StrRegion
from parser.tokenizer import Tokenizer
from parser.tokens import IdentNameToken, Token, DotToken, AttrNameToken


class TokenStreamFlag(IntFlag, boundary=FlagBoundary.STRICT):
    CONTENT = 1
    FULL = 2
    BOTH = CONTENT | FULL


EnumTV = TypeVar('EnumTV', bound=IntFlag)


def to_enum(obj: EnumTV | int | str, enum_t: type[EnumTV]) -> EnumTV:
    if isinstance(obj, str):
        return enum_t[obj]
    return enum_t(obj)


class MyTestCase(unittest.TestCase):
    def assertTokensEqual(
            self, t: Tokenizer, expected: Sequence[Token],
            stream: Union[str, int, TokenStreamFlag] = TokenStreamFlag.BOTH
    ):
        stream = to_enum(stream, TokenStreamFlag)
        assert stream
        if stream & TokenStreamFlag.CONTENT:
            self.assertEqual(t.content_tokens, expected)
        if stream & TokenStreamFlag.FULL:
            self.assertEqual(t.tokens, expected)

    def test__t_ident_name__at_end(self):
        t = Tokenizer('abc')
        end = t._t_ident_name(0)
        self.assertTokensEqual(t, [IdentNameToken(StrRegion(0, 3))])
        self.assertEqual(end, 3)
        t = Tokenizer('+c')
        end = t._t_ident_name(1)
        self.assertTokensEqual(t, [IdentNameToken(StrRegion(1, 2))])
        self.assertEqual(end, 2)

    def test__t_ident_name__normal(self):
        t = Tokenizer('ab+8')
        end = t._t_ident_name(0)
        self.assertTokensEqual(t, [IdentNameToken(StrRegion(0, 2))])
        self.assertEqual(end, 2)
        t = Tokenizer('2*ab.8')
        end = t._t_ident_name(2)
        self.assertTokensEqual(t, [IdentNameToken(StrRegion(2, 4))])
        self.assertEqual(end, 4)

    def test__t_attr_name__at_end(self):
        t = Tokenizer('e.0bc')
        end = t._t_ident_name(0)
        end = t._t_dot(end)
        end = t._t_attr_name(end)
        self.assertTokensEqual(t, [
            IdentNameToken(StrRegion(0, 1)),
            DotToken(StrRegion(1, 2)),
            AttrNameToken(StrRegion(2, 5))])
        self.assertEqual(end, 5)
        t = Tokenizer('-c')
        end = t._t_attr_name(1)
        self.assertTokensEqual(t, [AttrNameToken(StrRegion(1, 2))])
        self.assertEqual(end, 2)

    def test__t_attr_name__normal(self):
        t = Tokenizer('ab.c0+8')
        end = t._t_attr_name(3)
        self.assertTokensEqual(t, [AttrNameToken(StrRegion(3, 5))])
        self.assertEqual(end, 5)
        t = Tokenizer('2*ab.8.q')
        end = t._t_attr_name(5)
        self.assertTokensEqual(t, [AttrNameToken(StrRegion(5, 6))])
        self.assertEqual(end, 6)


if __name__ == '__main__':
    unittest.main()
