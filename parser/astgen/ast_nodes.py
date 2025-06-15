from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ast_node import AstNode, WalkerFnT

__all__ = [
    "AstNode", "AstProgramNode", "VarDeclScope", "VarDeclType", "AstDeclNode",
    "AstRepeat", "AstIf", "AstWhile", "AstAssign", "AstAugAssign", "AstDefine",
    "AstNumber", "AstString", "AstAnyName", "AstIdent", "AstAttrName",
    "AstListLiteral", "AstAttribute", "AstItem", "AstCall", "AstOp", "AstBinOp",
    "AstUnaryOp",
]


@dataclass
class AstProgramNode(AstNode):
    name = 'program'
    statements: list[AstNode]

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
class AstDeclNode(AstNode):
    name = 'var_decl'
    scope: VarDeclScope
    type: VarDeclType
    ident: AstIdent
    value: AstNode | None

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.ident, self.value))


@dataclass
class AstRepeat(AstNode):
    name = 'repeat'
    count: AstNode
    body: list[AstNode]

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.count, self.body))


@dataclass
class AstIf(AstNode):
    name = 'if'
    cond: AstNode
    if_body: list[AstNode]
    # elseif = else{if
    else_body: list[AstNode] | None = None
    # ^ Separate cases for no block and empty block (can be else {} to easily
    # add extra blocks in scratch interface)

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.cond, self.if_body, self.else_body))


@dataclass
class AstWhile(AstNode):
    name = 'while'
    cond: AstNode
    body: list[AstNode]

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.cond, self.body))


@dataclass
class AstAssign(AstNode):
    name = '='
    target: AstNode
    source: AstNode

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.target, self.source))


@dataclass
class AstAugAssign(AstNode):
    op: str  # maybe attach a StrRegion to the location of the op??
    target: AstNode
    source: AstNode

    @property
    def name(self):
        return self.op

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.target, self.source))


@dataclass
class AstDefine(AstNode):
    name = 'def'

    ident: AstIdent
    params: list[tuple[AstIdent, AstIdent]]  # type, ident
    body: list[AstNode]

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.ident, self.params, self.body))
# endregion ---- </Statements> ----


# region ---- <Expressions> ----
@dataclass
class AstNumber(AstNode):
    # No real point in storing the string representation (could always StrRegion.resolve())
    value: float | int


@dataclass
class AstString(AstNode):
    value: str  # Values with escapes, etc. resolved


@dataclass
class AstAnyName(AstNode):
    id: str

    def __post_init__(self):
        if type(self) == AstAnyName:
            raise TypeError("AstAnyName must not be instantiated directly.")


@dataclass
class AstIdent(AstAnyName):
    name = 'ident'


@dataclass
class AstAttrName(AstAnyName):
    name = 'attr'


@dataclass
class AstListLiteral(AstNode):
    name = 'list'
    items: list[AstNode]

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.items,))


@dataclass
class AstAttribute(AstNode):
    name = '.'
    obj: AstNode
    attr: AstAttrName

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.obj, self.attr))


@dataclass
class AstItem(AstNode):
    name = 'item'
    obj: AstNode
    index: AstNode

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.obj, self.index))


@dataclass
class AstCall(AstNode):
    name = 'call'
    obj: AstNode
    args: list[AstNode]

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.obj, self.args))


@dataclass
class AstOp(AstNode):
    op: str


@dataclass
class AstBinOp(AstOp):
    left: AstNode
    right: AstNode

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
class AstUnaryOp(AstOp):
    operand: AstNode

    valid_ops = ('+', '-', '!')

    def __post_init__(self):
        assert self.op in self.valid_ops

    @property
    def name(self):
        return self.op

    def _walk_members(self, fn: WalkerFnT):
        self.walk_multiple_objects(fn, (self.operand,))
# endregion ---- </Expressions> ----
