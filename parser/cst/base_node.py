from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from ..common import StrRegion, HasRegion

if TYPE_CHECKING:
    from ..tokens import Token


@dataclass
class Leaf(HasRegion):
    name: ClassVar[str]
    region: StrRegion
    parent: Node | None = None

    @classmethod
    def of(cls, token: Token, parent: Node | None = None):
        return cls(token.region, parent)


@dataclass
class Node(Leaf):
    children: list[AnyNode] = field(default_factory=list)

    @classmethod
    def of(cls, token: Token, parent: Node | None = None,
           children: list[AnyNode] | None = None):
        children = children or []
        return cls(token.region, parent, children)

    @classmethod  # Better args order
    def new(cls, region: StrRegion,
            children: list[AnyNode], parent: Node | None = None):
        return cls(region, parent, children)

    def __post_init__(self):
        children = self.children
        self.children = []
        self.add(*children)

    def add(self, *nodes: AnyNode, update_end=False):
        end = self.region.end
        for n in nodes:
            if n is None:
                raise TypeError("Cannot have `None` as child of AnyNode (for now??)")
            end = max(end, n.region.end)
            self.children.append(n)
            n.parent = self
        if update_end:
            self.region.end = end


AnyNode = Leaf
