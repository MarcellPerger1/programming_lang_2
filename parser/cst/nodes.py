from __future__ import annotations

from typing import cast

from util import checked_cast
from .named_node import (NamedLeafCls, NamedNodeCls, NamedSizedNodeCls,
                         register_corresponding_token)

__all__ = [  # Keep these sorted by category
    "ProgramNode", "AnyNullNode",
    # Atoms
    "NumberNode", "StringNode", "AnyNameLeaf", "IdentNode", "AttrNameNode",
    "AutocatNode",  # autocat is sorta atom-ish (it is in AST but not in CST?)
    # Item chains
    "GetattrNode", "GetitemNode", "ParenNode", "CallNode", "CallArgs",
    "ListNode",
    # Operators
    "OperatorNode",
    "UnaryOpNode", "UPlusNode", "UMinusNode", "NotNode",  # Unary operators
    # Binary operators
    "BinOpNode", "AddNode", "SubNode", "MulNode", "DivNode", "ModNode",
    "PowNode", "ConcatNode", "AndNode", "OrNode",
    # Comparisons
    "ComparisonNode", "EqNode", "NeqNode", "LtNode", "LeNode", "GtNode",
    "GeNode",

    "NopNode",
    # Blocks
    "BlockNode", "WhileBlock", "RepeatBlock",
    # If blocks & related infrastructure
    "ConditionalBlock", "IfBlock", "ElseIfBlock", "ElseBlock", "NullElseBlock",
    # Function Definitions
    "DefineNode", "ArgsDeclNode", "ArgDeclNode",
    # Variable declarations & related infrastructure
    "DeclNode", "DeclItemsList", "DeclItemNode",
    "DeclScopeNode", "DeclScope_Let", "DeclScope_Global",
    "DeclTypeNode", "DeclType_Variable", "DeclType_List",
    # Assignment (regular and augmented)
    "AssignOpNode", "AssignNode",
    "AddEqNode", "SubEqNode", "MulEqNode", "DivEqNode", "ModEqNode",
    "PowEqNode", "ConcatEqNode", "AndEqNode", "OrEqNode",
]


class ProgramNode(NamedNodeCls):
    name = 'program'  # Varargs

    @property
    def statements(self):
        return self.children


class AnyNullNode(NamedLeafCls):
    """For nodes whose presence denotes an absence of something in the syntax"""
    size = 0  # Name should be set by subclasses


# region ---- Expressions ----
@register_corresponding_token
class NumberNode(NamedLeafCls):
    name = 'number'


@register_corresponding_token
class StringNode(NamedLeafCls):
    name = 'string'


class AnyNameLeaf(NamedLeafCls):
    pass


@register_corresponding_token('ident_name')
class IdentNode(AnyNameLeaf):
    name = 'ident'


@register_corresponding_token('attr_name')
class AttrNameNode(AnyNameLeaf):
    name = 'attr'


class AutocatNode(NamedNodeCls):
    name = 'autocat'  # Note: this is varargs, unlike regular concat

    @property
    def parts(self) -> list[StringNode]:
        return cast(list[StringNode], self.children)


class GetattrNode(NamedSizedNodeCls):
    name = 'getattr'
    size = 2

    @property
    def target(self):
        return self.children[0]

    @property
    def attr(self):
        return checked_cast(AttrNameNode, self.children[1])


class GetitemNode(NamedSizedNodeCls):
    name = 'getitem'
    size = 2

    @property
    def target(self):
        return self.children[0]

    @property
    def item(self):
        return self.children[1]


class ParenNode(NamedSizedNodeCls):
    name = 'paren'
    size = 1

    @property
    def contents(self):
        return self.children[0]


class ListNode(NamedNodeCls):
    name = 'list'  # List literal, varargs

    @property
    def items(self):
        return self.children


class CallNode(NamedSizedNodeCls):
    name = 'call'
    size = 2

    @property
    def target(self):
        return self.children[0]

    @property
    def arglist(self) -> CallArgs:
        return checked_cast(CallArgs, self.children[1])


class CallArgs(NamedNodeCls):
    name = 'call_args'  # Varargs

    @property
    def args(self):
        return self.children


class OperatorNode(NamedNodeCls):
    pass


class UnaryOpNode(NamedSizedNodeCls, OperatorNode):
    size = 1

    @property
    def operand(self):
        return self.children[0]


@register_corresponding_token('+', arity=1)
@register_corresponding_token()
class UPlusNode(UnaryOpNode):
    name = '+(unary)'


@register_corresponding_token('-', arity=1)
@register_corresponding_token()
class UMinusNode(UnaryOpNode):
    name = '-(unary)'


@register_corresponding_token
class NotNode(UnaryOpNode):
    name = '!'


class BinOpNode(NamedSizedNodeCls, OperatorNode):
    size = 2

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]


@register_corresponding_token
class AddNode(BinOpNode):
    name = '+'


@register_corresponding_token
class SubNode(BinOpNode):
    name = '-'


@register_corresponding_token
class MulNode(BinOpNode):
    name = '*'


@register_corresponding_token
class DivNode(BinOpNode):
    name = '/'


@register_corresponding_token
class ModNode(BinOpNode):
    name = '%'


@register_corresponding_token
class PowNode(BinOpNode):
    name = '**'


@register_corresponding_token
class ConcatNode(BinOpNode):
    name = '..'


@register_corresponding_token
class AndNode(BinOpNode):
    name = '&&'


@register_corresponding_token
class OrNode(BinOpNode):
    name = '||'


class ComparisonNode(BinOpNode):
    pass


@register_corresponding_token
class EqNode(ComparisonNode):
    name = '=='


@register_corresponding_token
class NeqNode(ComparisonNode):
    name = '!='


@register_corresponding_token
class LtNode(ComparisonNode):
    name = '<'


@register_corresponding_token
class LeNode(ComparisonNode):
    name = '<='


@register_corresponding_token
class GtNode(ComparisonNode):
    name = '>'


@register_corresponding_token
class GeNode(ComparisonNode):
    name = '>='
# endregion


# region ---- Statements ----
class NopNode(NamedLeafCls):
    name = 'nop'


# region ---- Blocks ----
class BlockNode(NamedNodeCls):
    name = 'block'  # Varargs

    @property
    def statements(self):
        return self.children


class ConditionalBlock(NamedNodeCls):  # Varargs: if, *elseif, else
    name = 'conditional'

    @property
    def if_block(self):
        return checked_cast(IfBlock, self.children[0])

    @property
    def elseif_blocks(self):
        return cast(list[ElseIfBlock], self.children[1:-1])

    @property
    def else_block(self) -> ElseBlock | NullElseBlock:
        return checked_cast(ElseBlock | NullElseBlock, self.children[-1])


class IfBlock(NamedSizedNodeCls):
    name = 'if'
    size = 2  # condition, block

    @property
    def cond(self):
        return self.children[0]

    @property
    def block(self):
        return checked_cast(BlockNode, self.children[1])


class ElseIfBlock(NamedSizedNodeCls):
    name = 'elseif'
    size = 2

    @property
    def cond(self):
        return self.children[0]

    @property
    def block(self):
        return checked_cast(BlockNode, self.children[1])


class ElseBlock(NamedSizedNodeCls):
    name = 'else'
    size = 1  # just the BlockNode

    @property
    def block(self):
        return checked_cast(BlockNode, self.children[0])


class NullElseBlock(AnyNullNode):
    name = 'else_null'


class WhileBlock(NamedSizedNodeCls):
    name = 'while'
    size = 2

    @property
    def cond(self):
        return self.children[0]

    @property
    def block(self):
        return checked_cast(BlockNode, self.children[1])


class RepeatBlock(NamedSizedNodeCls):
    name = 'repeat'
    size = 2

    @property
    def count(self):
        return self.children[0]

    @property
    def block(self):
        return checked_cast(BlockNode, self.children[1])


class DefineNode(NamedSizedNodeCls):
    name = 'def'
    size = 3  # name, args_decl, block

    @property
    def ident(self):
        return checked_cast(IdentNode, self.children[0])

    @property
    def args_decl(self):
        return checked_cast(ArgsDeclNode, self.children[1])

    @property
    def block(self):
        return checked_cast(BlockNode, self.children[2])


class ArgsDeclNode(NamedNodeCls):
    name = 'args_decl'  # Varargs

    @property
    def decls(self):
        return cast(list[ArgDeclNode], self.children)


class ArgDeclNode(NamedSizedNodeCls):
    name = 'arg_decl'
    size = 2  # type and name

    @property
    def type(self):
        return checked_cast(IdentNode, self.children[0])

    @property
    def ident(self):
        return checked_cast(IdentNode, self.children[1])
# endregion


# region ---- Variable Decls ----
class DeclItemNode(NamedNodeCls):
    name = 'decl_item'  # 1 or 2 (name and optional value)

    @property
    def ident(self):
        return checked_cast(IdentNode, self.children[0])

    @property
    def value(self):
        if len(self.children) <= 1:
            return None
        return self.children[1]


class DeclScopeNode(AnyNullNode):
    pass


# noinspection PyPep8Naming
class DeclScope_Let(DeclScopeNode):
    name = 'scope__let'


# noinspection PyPep8Naming
class DeclScope_Global(DeclScopeNode):
    name = 'scope__global'


class DeclTypeNode(AnyNullNode):
    pass


# noinspection PyPep8Naming
class DeclType_Variable(DeclTypeNode):
    name = 'decl_type__variable'


# noinspection PyPep8Naming
class DeclType_List(DeclTypeNode):
    name = 'decl_type__list'


class DeclNode(NamedSizedNodeCls):
    name = 'var_decl'
    size = 3  # scope, type (value/list), decl_list

    @property
    def decl_scope(self):
        return checked_cast(DeclScopeNode, self.children[0])

    @property
    def decl_type(self):
        return checked_cast(DeclTypeNode, self.children[1])

    @property
    def decl_list(self):
        return cast(DeclItemsList, self.children[2])


class DeclItemsList(NamedNodeCls):
    name = 'decl_list'

    @property
    def decls(self):
        return cast(list[DeclItemNode], self.children)
# endregion


# region ---- Assignment-ops ----
class AssignOpNode(BinOpNode):
    @property
    def target(self):
        return self.children[0]

    @property
    def source(self):
        return self.children[1]


@register_corresponding_token
class AssignNode(AssignOpNode):
    name = '='


@register_corresponding_token
class AddEqNode(AssignOpNode):
    name = '+='


@register_corresponding_token
class SubEqNode(AssignOpNode):
    name = '-='


@register_corresponding_token
class MulEqNode(AssignOpNode):
    name = '*='


@register_corresponding_token
class DivEqNode(AssignOpNode):
    name = '/='


@register_corresponding_token
class ModEqNode(AssignOpNode):
    name = '%='


@register_corresponding_token
class PowEqNode(AssignOpNode):
    name = '**='


@register_corresponding_token
class ConcatEqNode(AssignOpNode):
    name = '..='


@register_corresponding_token
class AndEqNode(AssignOpNode):
    name = '&&='


@register_corresponding_token
class OrEqNode(AssignOpNode):
    name = '||='
# endregion
# endregion
