from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..str_region import StrRegion

if TYPE_CHECKING:
    from ..tokens import Token


# NOTE: these classes only follow the Liskov substitution principle on
# instances, not on __init__ and similar classmethods.

@dataclass
class Leaf:
    name: str
    region: StrRegion
    parent: Node | None = None

    @classmethod
    def of(cls, token: Token, name: str | None = '', parent: Node | None = None):
        return cls(name or token.name, token.region, parent)


@dataclass
class NamedLeafCls(Leaf):
    """A Leaf with a class-defined name"""
    name: str = field(init=False, repr=False)

    def __post_init__(self):
        if type(self) is NamedLeafCls:
            raise TypeError("NamedLeafCls may not be instantiated directly;"
                            " use a subclass or use Leaf")

    # noinspection PyMethodOverriding
    @classmethod
    def of(cls, token: Token, parent: Node | None = None):
        return cls(token.region, parent)


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


@dataclass
class NamedNodeCls(Node):
    """A Node with a class-defined name"""
    name: str = field(init=False, repr=False)

    def __post_init__(self):
        if type(self) is NamedNodeCls:
            raise TypeError("NamedNodeCls may not be instantiated directly;"
                            " use a subclass or use Node")
        super().__post_init__()

    # noinspection PyMethodOverriding
    @classmethod
    def of(cls, token: Token, children: list[AnyNode] | None = None,
           parent: Node | None = None):
        return cls(token.region, parent, children or [])

    # noinspection PyMethodOverriding
    @classmethod  # Better args order
    def new(cls, region: StrRegion, children: list[AnyNode], parent: Node = None):
        return cls(region, parent, children)


class NamedSizedNodeCls(NamedNodeCls):
    """A Node with a class-defined name and size."""
    size: int

    def __post_init__(self):
        # Don't call super().__post_init__() because we customise the _add() logic
        if type(self) is NamedSizedNodeCls:
            raise TypeError("NamedSizedNodeCls may not be instantiated directly;"
                            " use a subclass or use Node")
        children = self.children
        if len(children) != self.size:
            raise ValueError(f"{type(self).__name__} expected {self.size} "
                             f"children, got {len(children)}")
        self.children = []
        self._add(*children)

    def _add(self, *nodes: AnyNode, update_end=False):
        super().add(*nodes, update_end=update_end)

    def add(self, *nodes: AnyNode, update_end=False):
        raise TypeError(f"Cannot add nodes to fixed size {type(self).__name__}")


AnyNamedNode = NamedLeafCls | NamedNodeCls
