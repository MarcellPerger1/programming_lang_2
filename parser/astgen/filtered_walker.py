from __future__ import annotations

from typing import Callable, TypeVar

from util import flatten_force
from .ast_node import WalkerCallType, WalkableT, walk_ast

__all__ = ['WalkerFilterRegistry', 'FilteredWalker', 'walk_ast']

VT = TypeVar('VT')
WT = TypeVar('WT', bound=WalkableT)

SpecificCbT = Callable[[WT], bool | None]
SpecificCbsDict = dict[type[WT] | type, list[SpecificCbT[WT]]]
BothCbT = Callable[[WT, 'WalkerCallType'], bool | None]
BothCbsDict = dict[type[WT] | type, list[BothCbT[WT]]]


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

    def walk(self, o: WalkableT):
        return walk_ast(o, self)

    def __call__(self, o: WalkableT, call_type: WalkerCallType):
        result = None
        # Call more specific ones first
        specific_cbs = self.enter_cbs if call_type == WalkerCallType.PRE else self.exit_cbs
        for fn in self._get_funcs(specific_cbs, type(o)):
            if result := result or fn(o):
                return result  # Don't call later ones if already skipped
        for fn in self._get_funcs(self.both_cbs, type(o)):
            if result := result or fn(o, call_type):
                return result
        return result

    @classmethod
    def _get_funcs(cls, mapping: dict[type[WT] | type, list[VT]], tp: type[WT]) -> list[VT]:
        """Also looks at superclasses/MRO"""
        return flatten_force([mapping.get(sub, []) for sub in _get_mro(tp)])


def _get_mro(tp: type) -> tuple[type, ...]:  # tp.__mro__ but with proper types
    return tp.__mro__  # .mro() recalculates it every time, hence is slow
