from __future__ import annotations

from os import PathLike

from .simple_process_pool import *
from .timeouts import *


def readfile(path: int | str | bytes | PathLike[str] | PathLike[bytes],
             encoding='utf-8', errors: str | None = None,
             newline: str | None = None):
    with open(path, encoding=encoding, errors=errors, newline=newline, ) as f:
        return f.read()
