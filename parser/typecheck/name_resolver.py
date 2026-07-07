from __future__ import annotations

from util import assert_not_none
from .scope import NameInfo, FuncInfo, ParamInfo, Scope
from .types import ValType, ListType, BoolType, VoidType, PARAM_TYPES
from ..astgen.ast_node import AstNode
from ..astgen.ast_nodes import AstIdent, AstDeclNode, VarDeclScope, VarDeclType, AstDefine
from ..astgen.astgen import AstGen
from ..astgen.filtered_walker import FilteredWalker
from ..common import BaseLocatedError, RegionUnionArgT, region_union


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

    def run(self) -> Scope:
        if self.top_scope:
            return self.top_scope
        self._init()
        self.run_on_new_scope(self.ast.statements, curr_scope=self.top_scope)
        return assert_not_none(self.top_scope)

    def run_on_new_scope(self, block: list[AstNode], parent_scopes: list[Scope] = None,
                         curr_scope: Scope = None):
        # TODO: this could be done cleaner with a separate class for the walker
        def enter_ident(n: AstIdent):
            for s in reversed(scope_stack):
                if info := s.declared.get(n.id):
                    curr_scope.used[n.id] = info
                    return
            raise self.err(f"Name '{n.id}' is not defined", n)

        def enter_decl(n: AstDeclNode):
            # Need semi-special logic here to prevent walking it walking
            # the AstIdent that is currently being declared.
            AstNode.walk_obj(n.value, walker)  # Don't walk `n.ident`
            # Do this after walking (that is when the name is bound)
            ident = n.ident.id
            target_scope = curr_scope if n.scope == VarDeclScope.LET else self.top_scope
            if ident in target_scope.declared:
                raise self.err("Variable already declared", n.ident)
            target_scope.declared[ident] = NameInfo(target_scope, ident, (
                ValType() if n.type == VarDeclType.VARIABLE else ListType()))
            return True

        def enter_fn_decl(fn: AstDefine):
            ident = fn.ident.id
            if ident in curr_scope.declared:
                raise self.err("Function already declared", fn.ident)
            subscope = Scope()
            params: list[ParamInfo] = []
            for tp_node, name_node in fn.params:
                if tp_node.id not in PARAM_TYPES:
                    raise self.err("Unknown parameter type", tp_node)
                if (name := name_node.id) in subscope.declared:
                    raise self.err("There is already a parameter of this name", name_node)
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

    def err(self, msg: str, loc: RegionUnionArgT):
        return NameResolutionError(msg, region_union(loc), self.src)
