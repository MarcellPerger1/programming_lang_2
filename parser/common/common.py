from __future__ import annotations

from typing import TypeAlias, Sequence, Protocol, runtime_checkable

from .str_region import StrRegion

__all__ = ['HasRegion', 'RegionUnionArgT', 'region_union']


@runtime_checkable  # Eh, can't check subclass-ness but whatever
class HasRegion(Protocol):
    region: StrRegion


RegionUnionFlatT: TypeAlias = HasRegion | StrRegion | None
RegionUnionArgT: TypeAlias = RegionUnionFlatT | Sequence[RegionUnionFlatT]


def region_union(*args: RegionUnionArgT):
    regs = []
    for loc in args:
        if loc is None:
            continue
        if isinstance(loc, HasRegion):
            loc = loc.region
        if isinstance(loc, StrRegion):
            regs.append(loc)
        else:
            assert not isinstance(loc, str)
            regs.append(region_union(*loc))
    return StrRegion.union(*regs)
