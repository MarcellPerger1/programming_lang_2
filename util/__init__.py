from __future__ import annotations

from os import PathLike
from typing import TypeVar, Any, overload

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
