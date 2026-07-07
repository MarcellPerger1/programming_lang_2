from __future__ import annotations

import dataclasses
from os import PathLike
from typing import TypeVar, Any, overload, Iterable, Literal, TypeAlias

from .recursive_eq import recursive_eq
from .simple_process_pool import *
from .timeouts import *

T = TypeVar('T')
U = TypeVar('U')


def readfile(path: int | str | bytes | PathLike[str] | PathLike[bytes],
             encoding='utf-8', errors: str | None = None,
             newline: str | None = None):
    with open(path, encoding=encoding, errors=errors, newline=newline, ) as f:
        return f.read()


@overload
def checked_cast(typ: type[T], val: Any) -> T: ...
@overload
def checked_cast(typ: type[T | U], val: Any) -> T | U: ...


def checked_cast(typ: type[T], val: Any) -> T:
    assert isinstance(val, typ)
    return val


def flatten_force(seq: Iterable[Iterable[T]]) -> list[T]:
    return [item for sub in seq for item in sub]


def is_strict_subclass(o: object, type_or_types: tuple[type, ...]):
    try:
        types = tuple(type_or_types)
    except TypeError:
        types = (type_or_types,)
    return isinstance(o, type) and issubclass(o, types) and o not in types


DataclassesMissingT: TypeAlias = 'Literal[dataclasses._MISSING_TYPE.MISSING]'


# (Note: dataclasses._MISSING_TYPE isn't actually a runtime thing, it's just
# for type checkers to recognise dataclasses.MISSING)
def dcls_field_default(f: dataclasses.Field[T]) -> T | DataclassesMissingT:
    if f.default is not dataclasses.MISSING:
        return f.default
    if (factory := f.default_factory) is not dataclasses.MISSING:
        return factory()
    return dataclasses.MISSING


def assert_not_none(x: T | None) -> T:
    assert x is not None, "Expected non-None value, got None"
    return x
