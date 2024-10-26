from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..str_region import StrRegion

if TYPE_CHECKING:
    from ..tokens import Token


@dataclass
class Leaf:
    name: str
    region: StrRegion
    parent: Node | None = None

    @classmethod
    def of(cls, token: Token, name: str | None = '', parent: Node | None = None):
        return cls(name or token.name, token.region, parent)


AnyNode = Leaf


@dataclass
class Node(Leaf):
    children: list[AnyNode] = field(default_factory=list)

    @classmethod
    def of(cls, token: Token, name: str | None = '', parent: Node | None = None,
           children: list[AnyNode] | None = None):
        children = children or []
        return cls(name or token.name, token.region, parent, children)

    @classmethod  # Better args order
    def new(cls, name: str, region: StrRegion,
            children: list[AnyNode], parent: Node = None):
        return cls(name, region, parent, children)

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
            n.parent = n
        if update_end:
            self.region.end = end
