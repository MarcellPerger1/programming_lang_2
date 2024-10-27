from __future__ import annotations

from .tree_node import NamedLeafCls, NamedNodeCls, NamedSizedNodeCls


# region ---- Expressions ----
class NumberNode(NamedLeafCls):
    name = 'number'


class StringNode(NamedLeafCls):
    name = 'string'


class AnyNameLeaf(NamedLeafCls):
    pass


class IdentNode(AnyNameLeaf):
    name = 'ident'  # TODO actually ident_name


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
# endregion


# TODO: operators??!


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


class WhileBlock(NamedSizedNodeCls):
    name = 'while'
    size = 2


class RepeatBlock(NamedSizedNodeCls):
    name = 'repeat'
    size = 2
# endregion


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


class LetNode(NamedNodeCls):
    name = 'let_decl'  # Varargs


class GlobalNode(NamedNodeCls):
    name = 'global_decl'  # Varargs


class ProgramNode(NamedNodeCls):
    name = 'program'  # Varargs