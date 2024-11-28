from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar

from parser.astgen.ast_node import AstNode, walk_ast, WalkableT, WalkerCallType, AstIdent, \
    AstDeclNode, AstDefine, VarDeclType, VarDeclScope
from parser.astgen.astgen import AstGen
from parser.common import BaseLocatedError, StrRegion
from util import flatten_force

WT = TypeVar('WT', bound=WalkableT)
VT = TypeVar('VT')


class FilteredWalker:
    def __init__(self):
        self.enter_cbs: dict[type[WT] | type, list[Callable[[WT], bool | None]]] = {}
        self.exit_cbs: dict[type[WT] | type, list[Callable[[WT], bool | None]]] = {}
        self.both_cbs: dict[type[WT] | type, list[
            Callable[[WT, WalkerCallType], bool | None]]] = {}

    def register_both(self, t: type[WT], fn: Callable[[WT, WalkerCallType], bool | None]):
        self.both_cbs.setdefault(t, []).append(fn)
        return self

    def register_enter(self, t: type[WT], fn: Callable[[WT], bool | None]):
        self.enter_cbs.setdefault(t, []).append(fn)
        return self

    def register_exit(self, t: type[WT], fn: Callable[[WT], bool | None]):
        self.exit_cbs.setdefault(t, []).append(fn)
        return self

    def __call__(self, o: WalkableT, call_type: WalkerCallType):
        result = None
        # Call more specific ones first
        specific_cbs = self.enter_cbs if call_type == WalkerCallType.PRE else self.exit_cbs
        for fn in self._get_funcs(specific_cbs, type(o)):
            result = fn(o) or result
        for fn in self._get_funcs(self.both_cbs, type(o)):
            result = fn(o, call_type) or result
        return result

    @classmethod
    def _get_funcs(cls, mapping: dict[type[WT] | type, list[VT]], tp: type[WT]) -> list[VT]:
        """Also looks at superclasses/MRO"""
        return flatten_force(mapping.get(sub, []) for sub in tp.mro())


@dataclass
class TypeInfo:
    def __post_init__(self):
        assert type(self) != TypeInfo, "Cannot instantiate TypeInfo directly,use a subclass"


@dataclass
class ValType(TypeInfo):
    pass


@dataclass
class BoolType(TypeInfo):
    pass


@dataclass
class ListType(TypeInfo):
    pass


@dataclass
class VoidType(TypeInfo):
    """The ``void`` type - represents 'there must not be a value here'.

    For example, this is the return type of function that don't return anything
    (e.g. all regular user-defined scratch functions).
    """


@dataclass
class FunctionType(TypeInfo):
    arg_types: list[TypeInfo]
    ret_type: TypeInfo


@dataclass
class NameInfo:
    decl_scope: Scope
    ident: str
    tp_info: TypeInfo
    # node: AstNode  # <-- Why do we need this?
    is_param: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        assert type(self) != NameInfo, ("Cannot instantiate NameInfo directly,"
                                        " use a subclass or NameInfo.new()")


@dataclass
class FuncInfo(NameInfo):
    tp_info: FunctionType  # Overrides types (doesn't change order)
    # Can't just pass default_factory=Scope as it is only defined below
    subscope: Scope = field(default_factory=lambda: Scope())


@dataclass
class Scope:
    declared: dict[str, NameInfo] = field(default_factory=dict)
    used: dict[str, NameInfo] = field(default_factory=dict)
    # Add references to outer scopes' variables that we use


class NameResolutionError(BaseLocatedError):
    pass


# The reason `let` isn't used is because we don't want to imply similarity
#   between parameters as local variables (where none exists in Scratch).
#   Also, we might want to use `let` later as a modifier to bind it to
#     an actual local var.
# Don't need to `sys.intern` these manually as Python automatically does
#   this for literals.
PARAM_TYPES = {'number', 'string', 'val', 'bool'}


# Variables:
#  - We can prevent usages before the variable is declared in 2 ways:
#    - Based on time: very sensible, like JS, but requires too many runtime features
#    - Based on location: somewhat makes sense except for inner functions -
#       they may be called later so should be able to access any variables.
#  - Or we can just ignore it (e.g. `var` in JS) and pretend everything was
#     declared at the top (but not assigned to - i.e. hoist `var foo;` to top).
# To minimise accidental errors, option 1.2 is best
#  (errors shouldn't pass silently, and that method requires no special runtime)
class NameResolver:
    def __init__(self, astgen: AstGen):
        self.astgen = astgen
        self.src = self.astgen.src
        self.top_scope: Scope | None = None

    def _init(self):
        self.ast = self.astgen.parse()
        self.top_scope = Scope()

    def run(self):
        if self.top_scope:
            return self.top_scope
        self._init()
        self.run_on_new_scope(self.ast.statements, curr_scope=self.top_scope)
        return self.top_scope

    def run_on_new_scope(self, block: list[AstNode], scope_stack: list[Scope] = None,
                         curr_scope: Scope = None):
        def enter_ident(n: AstIdent):
            for s in scope_stack[::-1]:  # Inefficient, creates a copy!
                if info := s.declared.get(n.id):
                    curr_scope.used[n.id] = info
                    return
            raise self.err(f"Name '{n.id}' is not defined", n.region)

        def enter_decl(n: AstDeclNode):
            # Need semi-special logic here to prevent walking it walking
            # the AstIdent that is currently being declared.
            AstNode.walk_obj(n.value, walker)  # Don't walk `n.ident`
            # Do this after walking (that is when the name is bound)
            ident = n.ident.id
            target_scope = curr_scope if n.scope == VarDeclScope.LET else self.top_scope
            if ident in target_scope.declared:
                raise self.err("Variable already declared", n.region)
            target_scope.declared[ident] = NameInfo(target_scope, ident, (
                ValType() if n.type == VarDeclType.VARIABLE else ListType()))
            return True

        def enter_fn_decl(n: AstDefine):
            # Call this on enter (not evaluated right away and it the name can
            # be immediately used within the function i.e. recursion works)
            ident = n.ident.id
            if ident in curr_scope.declared:
                raise self.err("Function already declared", n.region)
            subscope = Scope()
            curr_scope.declared[ident] = info = FuncInfo(  # TODO this!
                curr_scope, ident, FunctionType(..., ...), subscope)
            for tp, param in n.params:
                if tp.id not in PARAM_TYPES:
                    raise self.err("Unknown parameter type", tp.region)
                if param.id in subscope.declared:
                    raise self.err("There is already a parameter of this name", param.region)
                subscope.declared[param.id] = NameInfo(
                    subscope, param.id, (
                        BoolType() if param.id == 'bool' else ValType()
                    ), is_param=True)
            # Skip walking body (only walking inner after we've collected
            # all the declared variables in current scope)
            inner_funcs.append((info, n))
            return True

        curr_scope = curr_scope or Scope()
        # Can't use `or` because need to preserve reference if arg is `[]`
        scope_stack = scope_stack if scope_stack is not None else []
        scope_stack.append(curr_scope)
        inner_funcs: list[tuple[FuncInfo, AstDefine]] = []
        # Walk self
        walker = (FilteredWalker()
                  .register_enter(AstIdent, enter_ident)
                  .register_enter(AstDeclNode, enter_decl)
                  .register_enter(AstDefine, enter_fn_decl))
        walk_ast(block, walker)
        # Walk sub-functions
        for fn_info, fn_decl in inner_funcs:
            fn_info.subscope = self.run_on_new_scope(fn_decl.body, scope_stack,
                                                     fn_info.subscope)
        return scope_stack.pop()  # Remove current scope from stack & return it

    def err(self, msg: str, region: StrRegion):
        return NameResolutionError(msg, region, self.src)
