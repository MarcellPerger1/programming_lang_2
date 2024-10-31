from typing import TypeAlias, Sequence

from .lexer.tokens import Token
from .str_region import StrRegion
from .trees.base_node import AnyNode

RegionUnionFlatT: TypeAlias = Token | AnyNode | StrRegion
RegionUnionArgT: TypeAlias = RegionUnionFlatT | Sequence[RegionUnionFlatT]


def region_union(*args: RegionUnionArgT):
    regs = []
    for loc in args:
        if isinstance(loc, (Token, AnyNode)):
            loc = loc.region  # Token and AnyNode, both have `.region`
        if isinstance(loc, StrRegion):
            regs.append(loc)
        else:
            assert not isinstance(loc, str)
            regs.append(region_union(*loc))
    return StrRegion.union(*regs)
