from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TypeMetadata:
    type: TypeInfo


@dataclass
class TypeInfo:
    def __post_init__(self):
        assert type(self) is not TypeInfo, (
            "Cannot instantiate TypeInfo directly, use a subclass")


@dataclass
class ValType(TypeInfo):
    def __str__(self):
        return 'val'


@dataclass
class BoolType(TypeInfo):
    def __str__(self):
        return 'bool'


@dataclass
class ListType(TypeInfo):
    def __str__(self):
        return 'list'


@dataclass
class VoidType(TypeInfo):
    """The ``void`` type - represents 'there must not be a value here'.

    For example, this is the return type of function that don't return anything
    (e.g. all regular user-defined scratch functions).
    """

    def __str__(self):
        return 'void'


@dataclass
class TypeType(TypeInfo):
    """The equivalent of Java's Class<T> or Python's type[T]"""
    tp: TypeInfo

    def __str__(self):
        # Now we come to the decision: how to display this: OCaml, Java/C++, or
        #  Python style? I'll just go with the standard angle brackets for now
        return f'type<{self.tp}>'


@dataclass
class FunctionType(TypeInfo):
    arg_types: list[TypeInfo]
    ret_type: TypeInfo

    def __str__(self):
        return f'({", ".join(map(str, self.arg_types))}) -> {self.ret_type}'


# The reason `let` isn't used is because we don't want to imply similarity
#   between parameters as local variables (where none exists in Scratch).
#   Also, we might want to use `let` later as a modifier to bind it to
#     an actual local var.
# Don't need to `sys.intern` these manually as Python automatically does
#   this for literals.
PARAM_TYPES = {'number', 'string', 'val', 'bool'}
