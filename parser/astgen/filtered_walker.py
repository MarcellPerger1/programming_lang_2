from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TypeVar, Generic

from util import flatten_force
from .ast_node import WalkerCallType, WalkableT, walk_ast

try:
    from typing import TypeVarTuple, Unpack, ParamSpec, Concatenate, Self
except ImportError:
    from typing_extensions import TypeVarTuple, Unpack, ParamSpec, Concatenate, Self


__all__ = ['WalkerFilterRegistry', 'FilteredWalker', 'walk_ast']

VT = TypeVar('VT')
WT = TypeVar('WT', bound=WalkableT)
WTT = TypeVar('WTT', bound=type)
Ps = TypeVarTuple('Ps')
Q = ParamSpec('Q')

SpecificCbT = Callable[[Unpack[Ps], WT], bool | None]
SpecificCbsDict = dict[type[WT], list[SpecificCbT[Unpack[Ps], WT]]]
# Equivalent to the above but requires passing type[WT] to first type arg
# This is done to workaround Pycharm bug PY-90860.
SpecificCbsDict3 = dict[WTT, list[SpecificCbT[Unpack[Ps], WT]]]

BothCbT = Callable[[Unpack[Ps], WT, WalkerCallType], bool | None]
BothCbsDict = dict[type[WT], list[BothCbT[Unpack[Ps], WT]]]
BothCbsDict3 = dict[WTT, list[BothCbT[Unpack[Ps], WT]]]  # Workaround to PY-90860


class WalkerFilterRegistry(Generic[Unpack[Ps], WT]):
    """Contains walkers over node types ``WT`` taking param types ``*Ps``
    followed by either ``(node: WT)`` for ``enter``/``exit`` or
    ``(node: WT, call_type: WalkerCallType)`` for ``both``. No walking
    functionality in and of itself due to the possibility of needing extra args"""
    def __init__(self,
                 enter_cbs: SpecificCbsDict3[type[WT], Unpack[Ps], WT] | None = None,
                 exit_cbs: SpecificCbsDict3[type[WT], Unpack[Ps], WT] | None = None,
                 both_sbc: BothCbsDict3[type[WT], Unpack[Ps], WT] | None = None):
        self.enter_cbs = dict(enter_cbs or {})  # Copy them
        self.exit_cbs = dict(exit_cbs or {})
        self.both_cbs = dict(both_sbc or {})

    def copy(self):
        return type(self)(self.enter_cbs, self.exit_cbs, self.both_cbs)

    def instantiate(self, *args: Unpack[Ps]):
        return BasicFilteredWalker[WT](
            self._instantiate_dict(self.enter_cbs, args),
            self._instantiate_dict(self.exit_cbs, args),
            self._instantiate_dict(self.both_cbs, args),
        )

    @classmethod
    def _instantiate_dict(
        cls,
        d: dict[WTT, list[Callable[Concatenate[Unpack[Ps], Q], bool | None]]],
        args: tuple[Unpack[Ps]]
    ) -> dict[WTT, list[Callable[Q, bool | None]]]:
        return {k: [functools.partial(f, *args) for f in fs] for k, fs in d.items()}

    def register_both(self, t: type[WT], fn: BothCbT[Unpack[Ps], WT]):
        self.both_cbs.setdefault(t, []).append(fn)
        return self

    def register_enter(self, t: type[WT], fn: SpecificCbT[Unpack[Ps], WT]):
        self.enter_cbs.setdefault(t, []).append(fn)
        return self

    def register_exit(self, t: type[WT], fn: SpecificCbT[Unpack[Ps], WT]):
        self.exit_cbs.setdefault(t, []).append(fn)
        return self

    def on_enter(self, *tps: type[WT]):
        """Decorator version of register_enter."""
        def decor(fn: SpecificCbT[Unpack[Ps], WT]):
            for t in tps:
                self.register_enter(t, fn)
            return fn
        return decor

    def on_exit(self, *tps: type[WT]):
        """Decorator version of register_exit."""
        def decor(fn: SpecificCbT[Unpack[Ps], WT]):
            for t in tps:
                self.register_exit(t, fn)
            return fn
        return decor

    def on_both(self, *tps: type[WT]):
        """Decorator version of register_both."""
        def decor(fn: BothCbT[Unpack[Ps], WT]):
            for t in tps:
                self.register_both(t, fn)
            return fn
        return decor


class BasicFilteredWalker(WalkerFilterRegistry[WT], Generic[WT]):
    def instantiate(self, *args: Unpack[()]):
        assert len(args) == 0, "Walker already instantiated"
        return self.copy()

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

    # `type[WT] | type` needed so we can access supertypes like `object` in the MRO
    @classmethod
    def _get_funcs(cls, mapping: dict[type[WT] | type, list[VT]], tp: type[WT]) -> list[VT]:
        """Also looks at superclasses/MRO"""
        return flatten_force([mapping.get(sub, []) for sub in _get_mro(tp)])


class FilteredWalker(BasicFilteredWalker[WT], Generic[WT]):
    def __init__(self):
        # Can then add studd to specific instances
        from_cls = (
            self.class_registry
            if isinstance(self.class_registry, WalkerFilterRegistry)
            else self.class_registry()
        ).instantiate(self)
        super().__init__(from_cls.enter_cbs, from_cls.exit_cbs, from_cls.both_cbs)

    @classmethod
    def class_registry(cls) -> WalkerFilterRegistry[Self, WT]:
        return WalkerFilterRegistry()

    @classmethod
    def create_cls_registry(cls, fn=None) -> WalkerFilterRegistry[Self, WT]:
        """Create a class-level registry that can be added to using decorators.

        This can be used in two ways (at the top of your class)::

            # MUST be this name  # Value can also be any WalkerFilterRegistry
            class_registry = FilteredWalker.create_cls_registry()

        or::

            @FilteredWalker.create_cls_registry
            @classmethod
            def class_registry(cls):  # MUST be this name
                pass  # Can return a WalkerFilterRegistry here

        and when registering methods::

            @class_registry.on_enter(AstDefine)
            def enter_define(self, ...):
                ...

        The restrictions on name are because we have no other way of detecting
         it (without metaclass dark magic) as we can't refer to the class while
         its namespace is being evaluated
        """
        if fn is None:
            return WalkerFilterRegistry()
        fn = fn.__func__ if isinstance(fn, classmethod) else fn
        if (parent := fn(cls)) is None:
            return WalkerFilterRegistry()
        return WalkerFilterRegistry.copy(parent)


def _get_mro(tp: type) -> tuple[type, ...]:  # tp.__mro__ but with proper types
    return tp.__mro__  # .mro() recalculates it every time, hence is slow
