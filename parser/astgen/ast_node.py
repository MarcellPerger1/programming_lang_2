from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeAlias, Iterable

from ..common import HasRegion, StrRegion

__all__ = ['AstNode', 'walk_ast', 'WalkableT', 'WalkerFnT', 'WalkerCallType',]


WalkableL0: TypeAlias = 'AstNode | list[AstNode] | tuple[AstNode, ...] | None'
WalkableT: TypeAlias = 'WalkableL0 | list[WalkableL0] | tuple[WalkableL0, ...]'
WalkerFnT: TypeAlias = Callable[[WalkableT, 'WalkerCallType'], bool | None]
"""Returns True if skip"""


class WalkerCallType(Enum):
    PRE = 'pre'
    POST = 'post'


@dataclass
class AstNode(HasRegion):
    region: StrRegion
    name = None  # type: str
    del name  # So we get better error msg if we forget to add it to a class

    def walk(self, fn: WalkerFnT):
        if fn(self, WalkerCallType.PRE):
            return
        self._walk_members(fn)
        fn(self, WalkerCallType.POST)

    def _walk_members(self, fn: WalkerFnT):
        """We have to define this manually on all subclasses with children.
        We don't try to do anything overcomplicated as it is hard to tell
        if a dataclass field is a child or not."""

    @classmethod
    def _walk_obj_members(cls, o: WalkableT, fn: WalkerFnT):
        if o is None:
            return
        if isinstance(o, AstNode):
            return o._walk_members(fn)
        try:
            it = iter(o)
        except TypeError:
            raise TypeError("Don't know how to walk object")
        for i in it:
            cls.walk_obj(i, fn)

    @classmethod
    def walk_obj(cls, o: WalkableT, fn: WalkerFnT):
        if isinstance(o, AstNode):
            return o.walk(fn)  # Delegate straight away (might have special functionality)
        if fn(o, WalkerCallType.PRE):
            return
        cls._walk_obj_members(o, fn)
        fn(o, WalkerCallType.POST)

    @classmethod
    def walk_multiple_objects(cls, fn: WalkerFnT, objs: Iterable[WalkableT]):
        for o in objs:
            cls.walk_obj(o, fn)


walk_ast = AstNode.walk_obj
