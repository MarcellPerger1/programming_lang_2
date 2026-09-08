from __future__ import annotations

from typing import TYPE_CHECKING, overload, Sequence, Literal, Sized

from util import checked_cast_class
from .base_node import Leaf, AnyNode, Node

if TYPE_CHECKING:
    from ..tokens import Token


# NOTE: these classes only follow the Liskov substitution principle on
# instances, not on __init__ and similar classmethods.


class SizedNode(Node):
    """A Node with a class-defined name and size."""
    size: int

    def __post_init__(self):
        # Don't call super().__post_init__() because we customise the _add() logic
        if type(self) is SizedNode:
            raise TypeError("NamedSizedNodeCls may not be instantiated "
                            "directly; use a subclass")
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


NAME_REGISTRY: dict[str | tuple[str, int], type[Node]] = {}


@overload
def register_corresponding_token(cls: type[Node], /, *,
                                 arity: int | Literal['auto'] | None = None): ...


@overload
def register_corresponding_token(*names: str, include_attr=False,
                                 arity: int | Literal['auto'] | None = None): ...


# Only really useful for thing with a 1-to-1 token-to-CST relation
# (atoms and operators mainly)
def register_corresponding_token(*args, include_attr=False,
                                 arity: int | Literal['auto'] | None = None):
    def register_once(name: str, cls: type[Node]):
        if arity is None:
            NAME_REGISTRY[name] = cls
        elif arity == 'auto':
            assert issubclass(cls, SizedNode)
            NAME_REGISTRY[name, cls.size] = cls
        else:
            NAME_REGISTRY[name, arity] = cls

    def decor(cls: type[Node]):
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


def node_cls_from_name(name: str, children: Sized | int | None = None,
                       arity: int | None = None) -> type[Leaf]:
    """Priority: arity > n_children > auto"""
    if arity is not None:
        return _cls_with_arity_or_general(name, arity)
    if children is not None:
        n_children = children if isinstance(children, int) else len(children)
        return _cls_with_arity_or_general(name, n_children)
    return NAME_REGISTRY[name]


def node_from_token(token: Token, children: Sequence[Node] | None = None,
                    parent: Node | None = None, arity: int | None = None):
    cls = node_cls_from_name(token.name, children, arity)
    if children:
        return checked_cast_class(Node, cls)(token.region, parent, list(children))
    return cls(token.region, parent)
