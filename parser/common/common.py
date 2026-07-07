from typing import TypeAlias, Sequence, cast, TYPE_CHECKING

from .str_region import StrRegion

if TYPE_CHECKING:
    from typing import TypeIs

__all__ = ['HasRegion', 'RegionUnionArgT', 'region_union']


class HasRegion:
    region: StrRegion

    @classmethod
    def has_instance_duck(cls, inst: object) -> TypeIs[HasRegion]:
        """Duck-typed version of the isinstance(inst, HasRegion) check"""
        return getattr(inst, 'region', None) is not None


RegionUnionFlatT: TypeAlias = HasRegion | StrRegion | None
RegionUnionArgT: TypeAlias = RegionUnionFlatT | Sequence[RegionUnionFlatT]


def region_union(*args: RegionUnionArgT):
    regs = []
    for loc in args:
        if loc is None:
            continue
        if HasRegion.has_instance_duck(loc):
            loc = loc.region
        if isinstance(loc, StrRegion):
            regs.append(loc)
        else:
            assert not isinstance(loc, str)
            regs.append(region_union(*loc))
    return StrRegion.union(*regs)
