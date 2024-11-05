from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..common import HasRegion, StrRegion


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
class VarDeclType(StrEnum):
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
# endregion ---- </Expressions> ----
