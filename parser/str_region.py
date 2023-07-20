from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrRegion:
    start: int
    end: int | None = None

    def resolve(self, s: str):
        if self.end is None:
            import warnings
            warnings.warn(RuntimeWarning(
                "Trying to call StrRegion.resolve with end=None."))
        return s[self.start:self.end]

    @classmethod
    def including(cls, *regions: StrRegion):
        """Returns smallest region that fits regions"""
        return StrRegion(min([r.start for r in regions]),
                         max([r.end for r in regions]))

    def union(self, *regions: StrRegion):
        return self.including(self, *regions)

    def __or__(self, other: StrRegion):
        return self.union(other)
