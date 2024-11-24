from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar

from parser.astgen.ast_node import AstNode, walk_ast, WalkableT, WalkerCallType, AstIdent, \
    AstDeclNode, AstDefine, VarDeclType
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


class NameType(Enum):
    VAR = 'var'
    LIST = 'list'
    FUNC = 'func'

    @classmethod
    def from_node(cls, n: AstNode):
        if isinstance(n, AstDefine):
            return cls.FUNC
        if not isinstance(n, AstDeclNode):
            raise TypeError(f"No corresponding NameType for '{n.name}' node")
        return cls.LIST if n.type == VarDeclType.LIST else cls.VAR


@dataclass
class NameInfo:
    scope: Scope
    ident: str
    # node: AstNode  # <-- Why do we need this?
    type = None  # type: NameType
    del type

    def __post_init__(self):
        assert type(self) != NameInfo, ("Cannot instantiate NameInfo directly,"
                                        " use a subclass or NameInfo.new()")


@dataclass
class FuncInfo(NameInfo):
    type = NameType.FUNC
    # Can't just pass default_factory=Scope as it is only defined below
    subscope: Scope = field(default_factory=lambda: Scope())


@dataclass
class VarInfo(NameInfo):
    type = NameType.VAR


@dataclass
class ListInfo(NameInfo):
    type = NameType.LIST


@dataclass
class Scope:
    declared: dict[str, NameInfo] = field(default_factory=dict)
    used: dict[str, NameInfo] = field(default_factory=dict)
    # Add references to outer scopes' variables that we use


class NameResolutionError(BaseLocatedError):
    pass


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

    def run(self):
        if self.top_scope:
            return self.top_scope
        self._init()
        self.top_scope = self.run_on_new_scope(self.ast.statements, scope_stack=[])
        return self.top_scope

    def run_on_new_scope(self, block: list[AstNode], scope_stack: list[Scope] = None):
        def enter_ident(n: AstIdent):
            for s in scope_stack[::-1]:  # Inefficient, creates a copy!
                if info := s.declared.get(n.id):
                    scope.used[n.id] = info
                    return
            raise self.err(f"Name '{n.id}' is not defined", n.region)

        def enter_decl(n: AstDeclNode):
            # Need semi-special logic here to prevent walking it walking
            # the AstIdent that is currently being declared.
            AstNode.walk_obj(n.value, walker)  # Don't walk `n.ident`
            # Do this after walking (that is when the name is bound)
            ident = n.ident.id
            if ident in scope.declared:
                raise self.err("Variable already declared", n.region)
            scope.declared[ident] = (
                VarInfo(scope, ident) if n.type == VarDeclType.VARIABLE
                else ListInfo(scope, ident))
            return True

        def enter_fn_decl(n: AstDefine):
            # Call this on enter (not evaluated right away and it the name can
            # be immediately used within the function i.e. recursion works)
            ident = n.ident.id
            if ident in scope.declared:
                raise self.err("Function already declared", n.region)
            scope.declared[ident] = info = FuncInfo(scope, ident)
            # Skip walking children (only walking inner after we've collected
            # all the declared variables in current scope)
            inner_funcs.append((info, n))
            return True

        scope = Scope()
        scope_stack = scope_stack if scope_stack is not None else []
        scope_stack.append(scope)
        inner_funcs: list[tuple[FuncInfo, AstDefine]] = []
        # Walk self
        walker = (FilteredWalker()
                  .register_enter(AstIdent, enter_ident)
                  .register_enter(AstDeclNode, enter_decl)
                  .register_enter(AstDefine, enter_fn_decl))
        walk_ast(block, walker)
        # Walk sub-functions
        for fn_info, fn_decl in inner_funcs:
            fn_info.subscope = self.run_on_new_scope(fn_decl.body, scope_stack)
        return scope_stack.pop()  # Remove current scope from stack & return it

    def err(self, msg: str, region: StrRegion):
        return NameResolutionError(msg, region, self.src)
