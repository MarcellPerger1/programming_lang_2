from typing import TypeAlias, Sequence

from .str_region import StrRegion


class HasRegion:
    region: StrRegion


RegionUnionFlatT: TypeAlias = HasRegion | StrRegion
RegionUnionArgT: TypeAlias = RegionUnionFlatT | Sequence[RegionUnionFlatT]


def region_union(*args: RegionUnionArgT):
    regs = []
    for loc in args:
        if getattr(loc, 'region', None) is not None:  # Duck-type HasRegion
            loc = loc.region
        if isinstance(loc, StrRegion):
            regs.append(loc)
        else:
            assert not isinstance(loc, str)
            regs.append(region_union(*loc))
    return StrRegion.union(*regs)
