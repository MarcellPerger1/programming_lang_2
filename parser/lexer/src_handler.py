from __future__ import annotations

from typing import Sequence

from .errors import LocatedTokenizerError
from .tokens import Token
from ..common import StrRegion, BaseLocatedError, region_union


class UsesSrc:
    def __init__(self, src: str):
        self.src: str = src

    def __getitem__(self, item: int | slice) -> str:
        return self.src[item]

    def eof(self, idx: int):
        return idx >= len(self.src)

    def get(self, idx: int, eof: str = '\0') -> str:
        try:
            return self.src[idx]
        except IndexError:
            return eof

    default_err_type = LocatedTokenizerError

    def err(self, msg: str,
            loc: int | Token | StrRegion | Sequence[int | Token | StrRegion],
            tp: type[BaseLocatedError] = None):
        try:
            seq: tuple[int | Token | StrRegion, ...] = tuple(loc)
        except TypeError:
            seq = (loc,)
        region = region_union([
            StrRegion(o, o + 1) if isinstance(o, int) else o
            for o in seq])
        tp = tp or self.default_err_type
        return tp(msg, region, self.src)
