"""Utils specific to this project. General utils that could be
used in all projects should go in utils.py"""
from __future__ import annotations

from enum import IntFlag, Enum
from typing import Sequence, TypeVar

from parser.astgen.ast_node import AstNode
from parser.astgen.astgen import AstGen
from parser.astgen.errors import LocatedAstError
from parser.common.error import BaseParseError
from parser.common.tree_print import tformat
from parser.cst.base_node import Leaf, AnyNode, Node
from parser.cst.cstgen import CstGen, LocatedCstError
from parser.lexer import Tokenizer
from parser.lexer.tokens import Token, OpToken
from parser.typecheck.typecheck import Scope, NameResolver, NameResolutionError
from test.common.snapshottest import SnapshotTestCase
from test.common.utils import TestCaseUtils
from util.pformat import pformat


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
        cls.format_dispatch.setdefault(AstNode, cls._tree_format)
        cls.format_dispatch.setdefault(Scope, pformat)
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

    def assertValidParseCST(self, src: str):
        self.assertIsNotNone(CstGen(Tokenizer(src)).parse())

    def assertFailsGracefullyCST(self, src: str):
        t = CstGen(Tokenizer(src))
        with self.assertRaises(LocatedCstError) as ctx:
            t.parse()
        return ctx.exception

    def assertNotInternalErrorCST(self, src: str):
        try:
            CstGen(Tokenizer(src)).parse()
        except BaseParseError:
            self.assertTrue(True)
        self.assertTrue(True)

    @classmethod
    def raiseInternalErrorsOnlyCST(cls, src: str):
        try:
            CstGen(Tokenizer(src)).parse()
        except BaseParseError:
            return None
        except Exception:
            raise
        return None

    def assertCstMatchesSnapshot(self, src: str):
        t = CstGen(Tokenizer(src))
        self.assertMatchesSnapshot(t.parse())

    def assertAstMatchesSnapshot(self, src: str):
        t = AstGen(CstGen(Tokenizer(src)))
        self.assertMatchesSnapshot(t.parse())

    def assertValidParseAST(self, src: str):
        self.assertIsNotNone(AstGen(CstGen(Tokenizer(src))).parse())

    def assertFailsGracefullyAST(self, src: str):
        a = AstGen(CstGen(Tokenizer(src)))
        with self.assertRaises(LocatedAstError) as ctx:
            a.parse()
        return ctx.exception

    # noinspection PyMethodMayBeStatic
    def getNameResolver(self, src: str):
        return NameResolver(AstGen(CstGen(Tokenizer(src))))

    def assertNameResolveError(self, src: str):
        nr = self.getNameResolver(src)
        with self.assertRaises(NameResolutionError) as ctx:
            nr.run()
        return ctx.exception
