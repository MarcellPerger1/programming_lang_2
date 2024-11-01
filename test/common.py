"""Utils specific to this project. General utils that could be
used in all projects should go in utils.py"""
from __future__ import annotations

from enum import IntFlag, Enum
from typing import Sequence, TypeVar

from parser.cst.base_node import Leaf, AnyNode, Node
from parser.tree_print import tformat
from parser.cst.treegen import TreeGen, CstParseError
from parser.error import BaseParseError
from parser.lexer import Tokenizer
from parser.lexer.tokens import Token, OpToken
from test.snapshottest import SnapshotTestCase
from test.utils import TestCaseUtils


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


EnumTV = TypeVar('EnumTV', bound=Enum)


def to_enum(obj: EnumTV | int | str, enum_t: type[EnumTV]) -> EnumTV:
    if isinstance(obj, str):
        return enum_t[obj]
    return enum_t(obj)


class CommonTestCase(SnapshotTestCase, TestCaseUtils):
    maxDiff = 65535

    @classmethod
    def _tree_format(cls, n: AnyNode):
        return tformat(n, verbose=True)

    @classmethod
    def setUpClass(cls) -> None:
        cls.format_dispatch.setdefault(Leaf, cls._tree_format)
        cls.format_dispatch.setdefault(Node, cls._tree_format)
        super().setUpClass()

    @classmethod
    def _token_as_tuple_no_region(cls, t: Token):
        if isinstance(t, OpToken):
            return t.name, t.op_str
        return (t.name, )

    def assertTokenStreamEquals(
            self, actual: Sequence[Token], expected: Sequence[Token],
            check_regions: bool = True):
        if check_regions:
            self.assertEqual(expected, actual)
        else:
            self.assertEqual([*map(self._token_as_tuple_no_region, expected)],
                             [*map(self._token_as_tuple_no_region, actual)])

    def assertTokensEqual(
            self, t: Tokenizer, expected: Sequence[Token],
            which_stream: str | int | TokenStreamFlag = TokenStreamFlag.BOTH,
            check_regions: bool = True
    ):
        stream = to_enum(which_stream, TokenStreamFlag)
        assert stream
        if stream & TokenStreamFlag.CONTENT:
            self.assertTokenStreamEquals(t.content_tokens, expected, check_regions)
        if stream & TokenStreamFlag.FULL:
            self.assertTokenStreamEquals(t.tokens, expected, check_regions)

    def assertValidParse(self, src: str):
        self.assertIsNotNone(TreeGen(Tokenizer(src)).parse())

    def assertFailsGracefully(self, src: str):
        t = TreeGen(Tokenizer(src))
        with self.assertRaises(CstParseError):
            t.parse()

    def assertNotInternalError(self, src: str):
        try:
            TreeGen(Tokenizer(src)).parse()
        except BaseParseError:
            self.assertTrue(True)
        self.assertTrue(True)

    @classmethod
    def raiseInternalErrorsOnly(cls, src: str):
        try:
            TreeGen(Tokenizer(src)).parse()
        except BaseParseError:
            return None
        except Exception:
            raise
        return None

    def assertTreeMatchesSnapshot(self, src: str):
        t = TreeGen(Tokenizer(src))
        self.assertMatchesSnapshot(t.parse())
