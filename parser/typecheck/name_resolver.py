from __future__ import annotations

from dataclasses import dataclass, field

from util.recursive_eq import recursive_eq
from ..astgen.ast_node import AstNode
from ..astgen.ast_nodes import AstIdent, AstDeclNode, VarDeclScope, VarDeclType, AstDefine
from ..astgen.astgen import AstGen
from ..astgen.filtered_walker import FilteredWalker

from ..common import BaseLocatedError, StrRegion
from .types import TypeInfo, FunctionType, ValType, ListType, BoolType, VoidType, PARAM_TYPES


@dataclass
class NameInfo:
    decl_scope: Scope
    ident: str
    tp_info: TypeInfo
    # node: AstNode  # <-- Why do we need this?
    is_param: bool = field(default=False, kw_only=True)


@dataclass
class FuncInfo(NameInfo):
    tp_info: FunctionType  # Overrides types (doesn't change order)
    params_info: list[ParamInfo]
    # Can't just pass default_factory=Scope as it is only defined below
    subscope: Scope = field(default_factory=lambda: Scope())

    @classmethod
    def from_param_info(
            cls, decl_scope: Scope, ident: str, params_info: list[ParamInfo],
            ret_type: TypeInfo, subscope: Scope = None):
        subscope = subscope or Scope()
        tp_info = FunctionType([p.tp for p in params_info], ret_type)
        return cls(decl_scope, ident, tp_info, params_info, subscope)


@dataclass
class ParamInfo:
    name: str
    tp: TypeInfo


@dataclass
class Scope:
    declared: dict[str, NameInfo] = field(default_factory=dict)
    used: dict[str, NameInfo] = field(default_factory=dict)
    """Add references to outer scopes' variables that we use.
    (so type codegen/type-checker knows what each AstIdent refers to)"""


# Prevent declared -> scope cmp recursion error
Scope.__eq__ = recursive_eq(Scope.__eq__)


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
        self.top_scope = Scope()

    def run(self):
        if self.top_scope:
            return self.top_scope
        self._init()
        self.run_on_new_scope(self.ast.statements, curr_scope=self.top_scope)
        return self.top_scope

    def run_on_new_scope(self, block: list[AstNode], parent_scopes: list[Scope] = None,
                         curr_scope: Scope = None):
        # TODO: this could be done cleaner with a separate class for the walker
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

        def enter_fn_decl(fn: AstDefine):
            ident = fn.ident.id
            if ident in curr_scope.declared:
                raise self.err("Function already declared", fn.ident.region)
            subscope = Scope()
            params: list[ParamInfo] = []
            for tp_node, name_node in fn.params:
                if tp_node.id not in PARAM_TYPES:
                    raise self.err("Unknown parameter type", tp_node.region)
                if (name := name_node.id) in subscope.declared:
                    raise self.err("There is already a parameter of this name",
                                   name_node.region)
                tp = BoolType() if tp_node.id == 'bool' else ValType()
                subscope.declared[name] = NameInfo(subscope, name, tp, is_param=True)
                params.append(ParamInfo(name, tp))
            curr_scope.declared[ident] = info = FuncInfo.from_param_info(
                curr_scope, ident, params,
                ret_type=VoidType(), subscope=subscope)
            inner_funcs.append((info, fn))  # Store funcs for later walking
            # Skip walking body, only walk inner after collecting all declared
            #  variables in outer scope so function can use all variables
            #  declared in outer scope - even the ones declared below it)
            return True

        curr_scope = curr_scope or Scope()
        scope_stack = parent_scopes or []
        scope_stack.append(curr_scope)
        inner_funcs: list[tuple[FuncInfo, AstDefine]] = []
        # Walk self
        walker = (FilteredWalker()
                  .register_enter(AstIdent, enter_ident)
                  .register_enter(AstDeclNode, enter_decl)
                  .register_enter(AstDefine, enter_fn_decl))
        walker.walk(block)
        # Walk sub-functions
        for fn_info, fn_decl in inner_funcs:
            fn_info.subscope = self.run_on_new_scope(
                fn_decl.body, scope_stack, fn_info.subscope)
        return scope_stack.pop()  # Remove current scope from stack & return it

    def err(self, msg: str, region: StrRegion):
        return NameResolutionError(msg, region, self.src)
