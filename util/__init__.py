from __future__ import annotations

from os import PathLike
from typing import TypeVar, Any, overload, Iterable

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
