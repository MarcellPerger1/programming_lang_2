from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ast_node import AstNode, WalkerFnT, MetadataT

__all__ = [
    "AstNode", "AstProgramNode", "VarDeclScope", "VarDeclType", "AstDeclNode",
    "AstRepeat", "AstIf", "AstWhile", "AstAssign", "AstAugAssign", "AstDefine",
    "AstNumber", "AstString", "AstAnyName", "AstIdent", "AstAttrName",
    "AstListLiteral", "AstAttribute", "AstItem", "AstCall", "AstOp", "AstBinOp",
    "AstUnaryOp",
]


@dataclass
class AstProgramNode(AstNode[MetadataT]):
    name = 'program'
    statements: list[AstNode[MetadataT]]

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.statements,))


# region ---- <Statements> ----
class VarDeclScope(Enum):
    LET = 'let'
    GLOBAL = 'global'


class VarDeclType(Enum):
    VARIABLE = 'variable'
    LIST = 'list'


@dataclass
class AstDeclNode(AstNode[MetadataT]):
    name = 'var_decl'
    scope: VarDeclScope
    type: VarDeclType
    ident: AstIdent
    value: AstNode[MetadataT] | None

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.ident, self.value))


@dataclass
class AstRepeat(AstNode[MetadataT]):
    name = 'repeat'
    count: AstNode[MetadataT]
    body: list[AstNode[MetadataT]]

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.count, self.body))


@dataclass
class AstIf(AstNode[MetadataT]):
    name = 'if'
    cond: AstNode[MetadataT]
    if_body: list[AstNode[MetadataT]]
    # elseif = else{if
    else_body: list[AstNode[MetadataT]] | None = None
    # ^ Separate cases for no block and empty block (can be else {} to easily
    # add extra blocks in scratch interface)

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.cond, self.if_body, self.else_body))


@dataclass
class AstWhile(AstNode[MetadataT]):
    name = 'while'
    cond: AstNode[MetadataT]
    body: list[AstNode[MetadataT]]

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.cond, self.body))


@dataclass
class AstAssign(AstNode[MetadataT]):
    name = '='
    target: AstNode[MetadataT]
    source: AstNode[MetadataT]

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.target, self.source))


@dataclass
class AstAugAssign(AstNode[MetadataT]):
    op: str  # maybe attach a StrRegion to the location of the op??
    target: AstNode[MetadataT]
    source: AstNode[MetadataT]

    @property
    def name(self):
        return self.op

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.target, self.source))


@dataclass
class AstDefine(AstNode[MetadataT]):
    name = 'def'

    ident: AstIdent
    # TODO: this should be list[AstDefineParam] where AstParam is an AstNode
    params: list[tuple[AstIdent, AstIdent]]  # type, ident
    body: list[AstNode[MetadataT]]

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.ident, self.params, self.body))
# endregion ---- </Statements> ----


# region ---- <Expressions> ----
@dataclass
class AstNumber(AstNode[MetadataT]):
    # No real point in storing the string representation (could always StrRegion.resolve())
    value: float | int


@dataclass
class AstString(AstNode[MetadataT]):
    value: str  # Values with escapes, etc. resolved


@dataclass
class AstAnyName(AstNode[MetadataT]):
    id: str

    def __post_init__(self):
        if type(self) == AstAnyName:
            raise TypeError("AstAnyName must not be instantiated directly.")


# TODO: AstIdent[MetadataT] here!!!
@dataclass
class AstIdent(AstAnyName):
    name = 'ident'


@dataclass
class AstAttrName(AstAnyName):
    name = 'attr'


@dataclass
class AstListLiteral(AstNode[MetadataT]):
    name = 'list'
    items: list[AstNode[MetadataT]]

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.items,))


@dataclass
class AstAttribute(AstNode[MetadataT]):
    name = '.'
    obj: AstNode[MetadataT]
    attr: AstAttrName

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.obj, self.attr))


@dataclass
class AstItem(AstNode[MetadataT]):
    name = 'item'
    obj: AstNode[MetadataT]
    index: AstNode[MetadataT]

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.obj, self.index))


@dataclass
class AstCall(AstNode[MetadataT]):
    name = 'call'
    obj: AstNode[MetadataT]
    args: list[AstNode[MetadataT]]

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.obj, self.args))


@dataclass
class AstOp(AstNode[MetadataT]):
    op: str


@dataclass
class AstBinOp(AstOp[MetadataT]):
    left: AstNode[MetadataT]
    right: AstNode[MetadataT]

    valid_ops = [*'+-*/%', '**', '..', '||', '&&',  # ops
                 '==', '!=', '<', '>', '<=', '>='  # comparisons
                 ]  # type: list[str]

    def __post_init__(self):
        assert self.op in self.valid_ops

    @property
    def name(self):
        return self.op

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.left, self.right))


@dataclass
class AstUnaryOp(AstOp[MetadataT]):
    operand: AstNode[MetadataT]

    valid_ops = ('+', '-', '!')

    def __post_init__(self):
        assert self.op in self.valid_ops

    @property
    def name(self):
        return self.op

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.operand,))
# endregion ---- </Expressions> ----
