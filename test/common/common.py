"""Utils specific to this project. General utils that could be
used in all projects should go in utils.py"""
from __future__ import annotations

import contextlib
from enum import IntFlag, Enum
from typing import Sequence, TypeVar, Any

from parser.astgen.ast_node import AstNode
from parser.astgen.astgen import AstGen
from parser.astgen.errors import LocatedAstError
from parser.astgen.filtered_walker import FilteredWalker
from parser.common import BaseLocatedError
from parser.common.error import BaseParseError
from parser.common.str_region import StrRegion
from parser.common.tree_print import tformat
from parser.cst.base_node import Leaf, AnyNode, Node
from parser.cst.cstgen import CstGen, LocatedCstError
from parser.lexer import Tokenizer
from parser.lexer.tokens import Token, OpToken
from parser.typecheck.name_resolver import NameResolutionError, NameResolver
from parser.typecheck.scope import Scope
from parser.typecheck.typecheck import Typechecker, TypecheckError
from parser.typecheck.types import TypeInfo, TypeMetadata
from util.pformat import pformat
from .snapshottest import SnapshotTestCase
from .utils import TestCaseUtils


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


T = TypeVar('T')
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
        return (t.name,)

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
    @contextlib.contextmanager
    def raiseInternalErrorsOnly(cls):
        try:
            yield
        except BaseParseError:
            pass
        except Exception:
            raise

    @classmethod
    def raiseInternalErrorsOnlyCST(cls, src: str):
        with cls.raiseInternalErrorsOnly():
            CstGen(Tokenizer(src)).parse()

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
    def getAstGen(self, src: str):
        return AstGen(CstGen(Tokenizer(src)))

    def getNameResolver(self, src: str):
        return NameResolver(self.getAstGen(src))

    def assertNameResolveError(self, src: str):
        nr = self.getNameResolver(src)
        with self.assertRaises(NameResolutionError) as ctx:
            nr.run()
        return ctx.exception

    def getTypechecker(self, src: str):
        return Typechecker(self.getNameResolver(src))

    def assertTypecheckedTo(self, node: AstNode[TypeMetadata] | None, expected: TypeInfo):
        node = self.assertAsNotNone(node, "Expected a node with metadata, got None")
        self.assertIsNotNone(node.meta, "Expected type metadata")
        self.assertEqual(expected, node.meta.type)

    def assertTypecheckError(self, src: str):
        tc = self.getTypechecker(src)
        with self.assertRaises(TypecheckError) as ctx:
            tc.run()
        return ctx.exception

    assertDoesntCrash = raiseInternalErrorsOnly

    def assertAllMetadata(self, n: AstNode[Any], expected_type: type[T]) -> AstNode[T]:
        def on_exit_node(nd: AstNode[Any]):
            if isinstance(nd, expected_type):
                return
            msg_extra = f"root={tformat(n)}\nnode={tformat(nd)}"
            self.assertIsNotNone(
                nd.meta, f"Expected node to have metadata:\n{msg_extra}")
            self.assertIsInstance(
                nd.meta, expected_type,
                f"Bad metadata type for node:\n{msg_extra}")

        # Report error with deepest one so on_exit
        FilteredWalker().register_exit(AstNode, on_exit_node).walk(n)
        return n

    def assertRegionEquals(self, expected: StrRegion, actual: StrRegion, src: str | None):
        if expected == actual:
            return
        if src:
            # Newlines added so that <lhs> != <rhs> output by unittest looks reasonable
            self.assertEqual(f'\n{expected.display(src)}\n',
                             f'\n{actual.display(src)}\n ',
                             "Expected regions to be equal (showing displayed)")
        self.assertEqual(expected, actual,  # Fallback in case display equal
                         "Expected regions to be equal (displayed as same)")

    def assertErrorRegion(self, expected: StrRegion, err: BaseLocatedError):
        self.assertRegionEquals(expected, err.region, err._src_text)
