from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, overload, Sequence, cast, Literal

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


# Types of fix Pycharm not understanding multiple inheritance w/ dataclasses
class _AnyNamedNodeT(AnyNode):
    name: str

    @overload
    def __init__(self, region: StrRegion, parent: Node | None = None): ...

    @overload
    def __init__(self, region: StrRegion, parent: Node | None = None,
                 children: Sequence[AnyNode] = ()): ...

    def __init__(self, region: StrRegion, parent: Node | None = None,
                 children: Sequence[AnyNode] = ()):
        self.region = region
        self.parent = parent
        self.children: list[AnyNode] = list(children)

    # noinspection PyMethodOverriding
    @classmethod
    def of(cls, token: Token, parent: Node | None = None):
        return cls(token.region, parent)


AnyNamedNode: type[_AnyNamedNodeT] | type[NamedLeafCls] = cast(
    type[_AnyNamedNodeT], NamedLeafCls)


@dataclass
class NamedNodeCls(Node, AnyNamedNode):
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


NAME_REGISTRY: dict[str | tuple[str, int], type[AnyNamedNode]] = {}


@overload
def register_corresponding_token(cls: type[AnyNamedNode],
                                 arity: int | Literal['auto'] | None = None): ...


@overload
def register_corresponding_token(*names: str, include_attr=False,
                                 arity: int | Literal['auto'] | None = None): ...


# Only really useful for thing with a 1-to-1 token-to-CST relation
# (atoms and operators mainly)
def register_corresponding_token(*args, include_attr=False,
                                 arity: int | Literal['auto'] = None):
    def register_once(name: str, cls: type[AnyNamedNode]):
        if arity is None:
            NAME_REGISTRY[name] = cls
        elif arity == 'auto':
            assert issubclass(cls, NamedSizedNodeCls)
            NAME_REGISTRY[name, cls.size] = cls
        else:
            # Cast required because Pycharm stupid
            NAME_REGISTRY[name, cast(int, arity)] = cls

    def decor(cls: type[AnyNamedNode]):
        for n in names:
            register_once(n, cls)
        if include_attr:
            register_once(cls.name, cls)
        return cls

    if len(args) == 1 and not isinstance(args[0], str):
        names = ()
        include_attr = True
        return decor(args[0])
    names = args
    return decor


def _cls_with_arity_or_general(name: str, arity: int):
    try:
        return NAME_REGISTRY[name, arity]
    except KeyError:
        return NAME_REGISTRY[name]


def node_cls_from_name(name: str, children: list | int | None = None, arity: int = None):
    """Priority: arity > n_children > auto"""
    if arity is not None:
        return _cls_with_arity_or_general(name, arity)
    if children is not None:
        n_children = children if isinstance(children, int) else len(children)
        return _cls_with_arity_or_general(name, n_children)
    return NAME_REGISTRY[name]


def node_from_token(token: Token, children: Sequence[Node] = None,
                    parent: Node | None = None, arity: int = None):
    cls = node_cls_from_name(token.name, children, arity)
    if children:
        return cls(token.region, parent, children)
    return cls(token.region, parent)
