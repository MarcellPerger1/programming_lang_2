from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Callable
from os import PathLike
from typing import TypeVar, overload, Literal, TYPE_CHECKING, Any

# re-export these
from .recursive_eq import recursive_eq
from .simple_process_pool import SimpleProcessPool
from .timeouts import *

if TYPE_CHECKING:
    # (Note: dataclasses._MISSING_TYPE isn't actually a runtime thing, it's just
    # for type checkers to recognise dataclasses.MISSING)
    DataclassesMissingT = Literal[dataclasses._MISSING_TYPE.MISSING]

T = TypeVar('T')
U = TypeVar('U')


def readfile(path: int | str | bytes | PathLike[str] | PathLike[bytes],
             encoding='utf-8', errors: str | None = None,
             newline: str | None = None):
    with open(path, encoding=encoding, errors=errors, newline=newline, ) as f:
        return f.read()


def checked_cast(typ: type[T], val: Any) -> T:
    assert isinstance(val, typ)
    return val


def checked_cast_class(as_subclass_of: type[T], cls: type) -> type[T]:
    assert issubclass(cls, as_subclass_of)
    return cls


def flatten_force(seq: Iterable[Iterable[T]]) -> list[T]:
    return [item for sub in seq for item in sub]


def is_strict_subclass(o: object, type_or_types: tuple[type, ...] | type) -> bool:
    types = tuple(pack_if_single_item(type_or_types))
    return isinstance(o, type) and issubclass(o, types) and o not in types


def dcls_field_default(f: dataclasses.Field[T]) -> T | DataclassesMissingT:
    if f.default is not dataclasses.MISSING:
        return f.default
    if (factory := f.default_factory) is not dataclasses.MISSING:
        return factory()
    return dataclasses.MISSING


def assert_not_none(x: T | None) -> T:
    assert x is not None, "Expected non-None value, got None"
    return x


def get_mro(t: type) -> tuple[type, ...]:  # T -> tuple[? super T, ...]
    return t.__mro__


@overload
def pack_if_single_item(iter_or_item: Iterable[T] | T,
                        ctor: Callable[[Iterable[T]], U]) -> U: ...


@overload
def pack_if_single_item(iter_or_item: Iterable[T] | T,
                        ctor: None = None) -> Iterable[T]: ...


def pack_if_single_item(iter_or_item: Iterable[T] | T,
                        ctor: Callable[[Iterable[T]], U] | None = None) -> U:
    try:
        it = iter(iter_or_item)
    except (TypeError, NotImplementedError):
        it = (iter_or_item, )
    return ctor(it) if ctor is not None else it
