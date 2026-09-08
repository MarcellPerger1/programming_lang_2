from __future__ import annotations

from dataclasses import dataclass, field

from util.recursive_eq import recursive_eq
from .types import TypeInfo, FunctionType


@dataclass
class NameInfo:
    decl_scope: Scope
    ident: str
    tp_info: TypeInfo
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
            ret_type: TypeInfo, subscope: Scope | None = None):
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
