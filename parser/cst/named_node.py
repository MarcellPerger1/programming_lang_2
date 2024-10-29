from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, overload, Sequence

from .base_node import Leaf, AnyNode, Node
from ..str_region import StrRegion

if TYPE_CHECKING:
    from ..tokens import Token


# NOTE: these classes only follow the Liskov substitution principle on
# instances, not on __init__ and similar classmethods.


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


@dataclass
class NamedNodeCls(Node, NamedLeafCls):
    """A Node with a class-defined name"""
    name: str = field(init=False, repr=False)

    def __post_init__(self):
        if type(self) is NamedNodeCls:
            raise TypeError("NamedNodeCls may not be instantiated directly;"
                            " use a subclass or use Node")
        Node.__post_init__(self)

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


AnyNamedNode = NamedLeafCls


NAME_REGISTRY: dict[str, type[AnyNamedNode]] = {}


@overload
def register_node_cls(cls: type[AnyNamedNode]): ...
@overload
def register_node_cls(*names: str, include_attr=False): ...


def register_node_cls(*args, include_attr=False):
    def decor(cls: type[AnyNamedNode]):
        for n in names:
            NAME_REGISTRY[n] = cls
        if include_attr:
            NAME_REGISTRY[cls.name] = cls
        return cls

    if len(args) == 1 and not isinstance(args[0], str):
        names = ()
        include_attr = True
        return decor(args[0])
    names = args
    return decor


# (Pycharm doesn't understand storing classes, probably because it's made in Java ;)
# noinspection PyArgumentList
def node_from_token(token: Token, children: Sequence[Node] = (), parent: Node | None = None):
    if children:
        return NAME_REGISTRY[token.name](token.region, parent, children)
    return NAME_REGISTRY[token.name](token.region, parent)

