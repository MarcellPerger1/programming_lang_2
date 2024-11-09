from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..common import HasRegion, StrRegion

__all__ = [
    "AstNode", "AstProgramNode", "VarDeclType", "AstDeclNode", "AstRepeat",
    "AstIf", "AstWhile", "AstAssign", "AstAugAssign", "AstDefine", "AstNumber",
    "AstString", "AstAnyName", "AstIdent", "AstAttrName", "AstAttribute",
    "AstItem", "AstCall", "AstOp", "AstBinOp", "AstUnaryOp",
]


@dataclass
class AstNode(HasRegion):
    region: StrRegion
    name = None  # type: str
    del name  # So we get better error msg if we forget to add it to a class


@dataclass
class AstProgramNode(AstNode):
    name = 'program'
    statements: list[AstNode]


# region ---- <Statements> ----
class VarDeclType(Enum):
    LET = 'let'
    GLOBAL = 'global'


@dataclass
class AstDeclNode(AstNode):
    name = 'var_decl'
    type: VarDeclType
    decls: list[tuple[AstIdent, AstNode | None]]


@dataclass
class AstRepeat(AstNode):
    name = 'repeat'
    count: AstNode
    body: list[AstNode]


@dataclass
class AstIf(AstNode):
    name = 'if'
    cond: AstNode
    if_body: list[AstNode]
    # elseif = else{if
    else_body: list[AstNode] | None = None
    # ^ Separate cases for no block and empty block (can be else {} to easily
    # add extra blocks in scratch interface)


@dataclass
class AstWhile(AstNode):
    name = 'while'
    cond: AstNode
    body: list[AstNode]


@dataclass
class AstAssign(AstNode):
    name = '='
    target: AstNode
    source: AstNode


@dataclass
class AstAugAssign(AstNode):
    op: str  # maybe attach a StrRegion to the location of the op??
    target: AstNode
    source: AstNode

    @property
    def name(self):
        return self.op


@dataclass
class AstDefine(AstNode):
    name = 'def'

    ident: AstIdent
    params: list[tuple[AstIdent, AstIdent]]  # type, ident
    body: list[AstNode]
# endregion ---- </Statements> ----


# region ---- <Expressions> ----
@dataclass
class AstNumber(AstNode):
    value: float  # No real point in storing the string representation (can just


@dataclass
class AstString(AstNode):
    value: str


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
class AstAttribute(AstNode):
    name = '.'
    obj: AstNode
    attr: AstAttrName


@dataclass
class AstItem(AstNode):
    name = 'item'
    obj: AstNode
    index: AstNode


@dataclass
class AstCall(AstNode):
    name = 'call'
    obj: AstNode
    args: list[AstNode]


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


@dataclass
class AstUnaryOp(AstOp):
    operand: AstNode

    valid_ops = ('+', '-', '!')

    def __post_init__(self):
        assert self.op in self.valid_ops

    @property
    def name(self):
        return self.op
# endregion ---- </Expressions> ----
