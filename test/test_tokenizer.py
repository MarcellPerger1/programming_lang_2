import unittest
from enum import IntFlag
from typing import Sequence, TypeVar, Union

from parser.lexer.tokens import WhitespaceToken, StringToken, EofToken, NumberToken
from parser.str_region import StrRegion
from parser.lexer import Tokenizer
from parser.tokens import IdentNameToken, Token, DotToken, AttrNameToken, OpToken


def _strict_boundary_kwargs():
    try:
        from enum import FlagBoundary
        return {'boundary': FlagBoundary.STRICT}
    except ImportError:
        return {}  # Python 3.10


class TokenStreamFlag(IntFlag, **_strict_boundary_kwargs()):
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

    def test__t_number(self):
        t = Tokenizer('a+432_123+e')
        end = t._t_number(2)
        self.assertTokensEqual(t, [NumberToken(StrRegion(2, 9))])
        self.assertEqual(end, 9)
        t = Tokenizer('2.')
        end = t._t_number(0)
        self.assertTokensEqual(t, [NumberToken(StrRegion(0, 2))])
        self.assertEqual(end, 2)

    def test_mod_supported(self):
        t = Tokenizer('a+b%2')
        t.tokenize()
        self.assertTokensEqual(t, [
            IdentNameToken(StrRegion(0, 1)),
            OpToken(StrRegion(1, 2), '+'),
            IdentNameToken(StrRegion(2, 3)),
            OpToken(StrRegion(3, 4), '%'),
            NumberToken(StrRegion(4, 5)),
            EofToken(StrRegion(5, 5))
        ])

    def test_tokenize_concat_works(self):
        t = Tokenizer('ab .. "s"')
        t.tokenize()
        self.assertTokensEqual(t, [
            IdentNameToken(StrRegion(0, 2)),
            WhitespaceToken(StrRegion(2, 3)),
            OpToken(StrRegion(3, 5), '..'),
            WhitespaceToken(StrRegion(5, 6)),
            StringToken(StrRegion(6, 9)),
            EofToken(StrRegion(9, 9)),
        ], TokenStreamFlag.FULL)
        self.assertTokensEqual(t, [
            IdentNameToken(StrRegion(0, 2)),
            OpToken(StrRegion(3, 5), '..'),
            StringToken(StrRegion(6, 9)),
            EofToken(StrRegion(9, 9)),
        ], TokenStreamFlag.CONTENT)


if __name__ == '__main__':
    unittest.main()
