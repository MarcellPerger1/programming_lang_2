from __future__ import annotations

from .named_node import NamedLeafCls, NamedNodeCls, NamedSizedNodeCls, register_corresponding_token

__all__ = [
    "NumberNode", "StringNode", "AnyNameLeaf", "IdentNode", "AttrNameNode",
    "AutocatNode", "GetattrNode", "GetitemNode", "ParenNode", "CallNode",
    "CallArgs", "NopNode", "BlockNode", "ConditionalBlock", "IfBlock",
    "ElseIfBlock", "ElseBlock", "NullElseBlock", "WhileBlock", "RepeatBlock",
    "DefineNode", "ArgsDeclNode", "ArgDeclNode", "DeclItemNode", "LetNode",
    "GlobalNode", "ProgramNode",
]


class ProgramNode(NamedNodeCls):
    name = 'program'  # Varargs


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
    name = 'ident'  # TODO actually ident_name


@register_corresponding_token
class AttrNameNode(AnyNameLeaf):
    name = 'attr_name'


class AutocatNode(NamedNodeCls):
    name = 'autocat'  # Note: this is varargs, unlike regular concat


class GetattrNode(NamedSizedNodeCls):
    name = 'getattr'
    size = 2


class GetitemNode(NamedSizedNodeCls):
    name = 'getitem'
    size = 2


class ParenNode(NamedSizedNodeCls):
    name = 'paren'
    size = 1
    in_ast = False  # TODO Not used yet


class CallNode(NamedSizedNodeCls):
    name = 'call'
    size = 2


class CallArgs(NamedNodeCls):
    name = 'call_args'  # Varargs


class OperatorNode(NamedNodeCls):
    pass


class UnaryOpNode(NamedSizedNodeCls, OperatorNode):
    size = 1


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
    name = 'nop'  # TODO: actually nop_smt


# region ---- Blocks ----
class BlockNode(NamedNodeCls):
    name = 'block'  # Varargs


class ConditionalBlock(NamedNodeCls):  # Varargs: if, *elseif, else
    name = 'if'  # TODO maybe conditional


class IfBlock(NamedSizedNodeCls):
    name = 'if_cond'  # TODO maybe just if
    size = 2  # condition, block


class ElseIfBlock(NamedSizedNodeCls):
    name = 'elseif_cond'  # TODO maybe elseif
    size = 2


class ElseBlock(NamedSizedNodeCls):
    name = 'else_cond'  # TODO maybe just else
    size = 1  # just the BlockNode


class NullElseBlock(NamedSizedNodeCls):
    name = 'else_cond_NULL'  # TODO maybe just else_null
    size = 0


class WhileBlock(NamedSizedNodeCls):
    name = 'while'
    size = 2


class RepeatBlock(NamedSizedNodeCls):
    name = 'repeat'
    size = 2


class DefineNode(NamedSizedNodeCls):
    name = 'def'
    size = 3  # name, args_decl, block


class ArgsDeclNode(NamedNodeCls):
    name = 'args_decl'  # Varargs


class ArgDeclNode(NamedSizedNodeCls):
    name = 'arg_decl'
    size = 2  # type and name


class DeclItemNode(NamedNodeCls):
    name = 'decl_item'  # 1 or 2 (name and optional value)
# endregion


class LetNode(NamedNodeCls):
    name = 'let_decl'  # Varargs


class GlobalNode(NamedNodeCls):
    name = 'global_decl'  # Varargs


# region ---- Assignment-ops ----
class AssignOpNode(BinOpNode):
    pass


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
