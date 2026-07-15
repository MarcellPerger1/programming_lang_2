from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias, Iterable, TypeVar, TYPE_CHECKING, Generic

from ..common import HasRegion, StrRegion

__all__ = ['AstNode', 'walk_ast', 'WalkableT', 'WalkerFnT', 'WalkerCallType', 'MetadataT']


# noinspection PyUnreachableCode
# If we are a type checker or a new enough runtime, we want default
if TYPE_CHECKING or sys.version_info >= (3, 13):
    # If we are an old version, import it (for type checkers; unreachable at runtime)
    if sys.version_info < (3, 13):
        from typing_extensions import TypeVar
    # Now we have a default-able TypeVar from somewhere so can use it
    MetadataT = TypeVar('MetadataT', default=None)
else:
    # We are an old runtime version without default
    MetadataT = TypeVar('MetadataT')


class WalkerCallType(Enum):
    PRE = 'pre'
    POST = 'post'


@dataclass
class AstNode(HasRegion, Generic[MetadataT]):
    region: StrRegion
    name = None  # type: str
    del name  # So we get better error msg if we forget to add it to a class
    meta: MetadataT = field(kw_only=True, default=None)

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


WalkableL0: TypeAlias = AstNode | Sequence[AstNode] | None
WalkableT: TypeAlias = WalkableL0 | Sequence[WalkableL0]
WalkerFnT: TypeAlias = Callable[[WalkableT, WalkerCallType], bool | None]
"""Returns True if skip"""
