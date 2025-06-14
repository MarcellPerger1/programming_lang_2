from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeAlias, Iterable, TypeVar

from util import flatten_force
from ..common import HasRegion, StrRegion

__all__ = ['AstNode', 'walk_ast', 'WalkableT', 'WalkerFnT', 'WalkerCallType',
           "FilteredWalker"]

VT = TypeVar('VT')
WT = TypeVar('WT', bound='WalkableT')

WalkableL0: TypeAlias = 'AstNode | list[AstNode] | tuple[AstNode, ...] | None'
WalkableT: TypeAlias = 'WalkableL0 | list[WalkableL0] | tuple[WalkableL0, ...]'
WalkerFnT: TypeAlias = Callable[[WalkableT, 'WalkerCallType'], bool | None]
"""Returns True if skip"""

SpecificCbT = Callable[[WT], bool | None]
SpecificCbsDict = dict[type[WT] | type, list[Callable[[WT], bool | None]]]
BothCbT = Callable[[WT, 'WalkerCallType'], bool | None]
BothCbsDict = dict[type[WT] | type, list[Callable[[WT, 'WalkerCallType'], bool | None]]]


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


class WalkerFilterRegistry:
    def __init__(self, enter_cbs: SpecificCbsDict = (),
                 exit_cbs: SpecificCbsDict = (),
                 both_sbc: BothCbsDict = ()):
        self.enter_cbs: SpecificCbsDict = dict(enter_cbs)  # Copy them,
        self.exit_cbs: SpecificCbsDict = dict(exit_cbs)    # also converts default () -> {}
        self.both_cbs: BothCbsDict = dict(both_sbc)

    def copy(self):
        return WalkerFilterRegistry(self.enter_cbs, self.exit_cbs, self.both_cbs)

    def register_both(self, t: type[WT], fn: Callable[[WT, WalkerCallType], bool | None]):
        self.both_cbs.setdefault(t, []).append(fn)
        return self

    def register_enter(self, t: type[WT], fn: Callable[[WT], bool | None]):
        self.enter_cbs.setdefault(t, []).append(fn)
        return self

    def register_exit(self, t: type[WT], fn: Callable[[WT], bool | None]):
        self.exit_cbs.setdefault(t, []).append(fn)
        return self

    def __call__(self, *args, **kwargs):
        return self

    def on_enter(self, *tps: type[WT] | type):
        """Decorator version of register_enter."""
        def decor(fn: SpecificCbT):
            for t in tps:
                self.register_enter(t, fn)
            return fn
        return decor

    def on_exit(self, *tps: type[WT] | type):
        """Decorator version of register_exit."""
        def decor(fn: SpecificCbT):
            for t in tps:
                self.register_exit(t, fn)
            return fn
        return decor

    def on_both(self, *tps: type[WT] | type):
        """Decorator version of register_both."""
        def decor(fn: BothCbT):
            for t in tps:
                self.register_both(t, fn)
            return fn
        return decor


class FilteredWalker(WalkerFilterRegistry):
    def __init__(self):
        cls_reg = self.class_registry()
        super().__init__(cls_reg.enter_cbs, cls_reg.exit_cbs, cls_reg.both_cbs)

    @classmethod
    def class_registry(cls) -> WalkerFilterRegistry:
        return WalkerFilterRegistry()

    @classmethod
    def create_cls_registry(cls, fn=None):
        """Create a class-level registry that can be added to using decorators.

        This can be used in two ways (at the top of your class)::

            # MUST be this name
            class_registry = FilteredWalker.create_cls_registry()

        or::

            @classmethod
            @FilteredWalker.create_cls_registry
            def class_registry(cls):  # MUST be this name
                pass

        and when registering methods::

            @class_registry.on_enter(AstDefine)
            def enter_define(self, ...):
                ...

        The restrictions on name are because we have no other way of detecting
         it (without metaclass dark magic) as we can't refer to the class while
         its namespace is being evaluated
        """
        if fn is not None and (parent := fn(cls)) is not None:
            return WalkerFilterRegistry.copy(parent)
        return WalkerFilterRegistry()

    def __call__(self, o: WalkableT, call_type: WalkerCallType):
        result = None
        # Call more specific ones first
        specific_cbs = self.enter_cbs if call_type == WalkerCallType.PRE else self.exit_cbs
        for fn in self._get_funcs(specific_cbs, type(o)):
            result = fn(o) or result
        for fn in self._get_funcs(self.both_cbs, type(o)):
            result = fn(o, call_type) or result
        return result

    @classmethod
    def _get_funcs(cls, mapping: dict[type[WT] | type, list[VT]], tp: type[WT]) -> list[VT]:
        """Also looks at superclasses/MRO"""
        return flatten_force(mapping.get(sub, []) for sub in tp.mro())
# endregion
