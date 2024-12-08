"""A better version of the builtin ``pprint` that doesn't use way too much
horizontal space for dataclasses and is generally much more sensible, e
specially for highly recursive data structures and dataclasses.

An additional restriction that this tries to satisfy is that
``pformat(a) == pformat(b)`` if an only if ``a == b``
(subject to a sensible ``__eq__``/``__repr__``/etc.)"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import sys
from dataclasses import dataclass
from typing import Sequence, Any, IO

__all__ = ['PrettyFormatter', 'pformat', 'pprint']


def pformat(o: object, indent: int = 2, max_simple_len: int = 64):
    return PrettyFormatter(indent, max_simple_len).format(o)


def pprint(o: object, stream: IO[str] = None, indent: int = 2, max_simple_len: int = 64):
    return PrettyFormatter(indent, max_simple_len).print(o, stream)


class PrettyFormatter:
    leaf_types: set[type] = {
        int, float, str, complex, bool, type(None), type(NotImplemented),
        enum.Enum
    }

    def __init__(self, indent=2, max_simple_len: int = 64):
        self.indent = indent
        self.max_simple_len = max_simple_len

    def print(self, o: object, stream: IO[str] = None):
        # TODO: this could be optimised by just passing a custom StreamDest
        #  that just delegates .write() to underlying stream
        stream = stream or sys.stdout
        stream.write(self.format(o))

    def format(self, o: object):
        dest = Dest()
        self._fmt(dest, 0, ContextStack(), o)
        return dest.get_value()

    def _fmt(self, dest: Dest, indent: int, ctx: ContextStack, o: object):
        # Handle internal hacks (to make _get_seq work) before the entering
        # the ctx as we don't want our internals showing up on the stack
        # and ruining the `up` values in circular refs
        if isinstance(o, DontReprMe):
            return dest.write(o.string)  # Don't write indent - handled by parent
        if isinstance(o, StringJoin):
            return self._fmt_string_join(dest, indent, ctx, o)
        if (idx := ctx.get_idx(o)) is not None:
            return self._fmt_circular(dest, indent, ctx, idx)
        with ctx.enter_context(o):
            return self._fmt_inner(dest, indent, ctx, o)

    def _fmt_string_join(self, dest: Dest, indent: int, ctx: ContextStack, join: StringJoin):
        # Don't write indent on first line (handled by parent)
        for v in join.parts:
            # Only indent first thing (done above)
            self._fmt(dest, indent, ctx, v)
        return

    def _fmt_inner(self, dest: Dest, indent: int, ctx: ContextStack, o: object):
        if isinstance(o, enum.Enum):
            # Don't write indent on first line (handled by parent)
            return dest.write(f'{type(o).__name__}.{o.name}')
        if isinstance(o, list):
            return self._fmt_seq(dest, indent, ctx, o, '[', ']')
        if isinstance(o, tuple):
            return self._fmt_seq(dest, indent, ctx, o, '(', ')',
                                 trailing_comma_if_unitary=True)
        if isinstance(o, set):
            if len(o) == 0:
                return dest.write('set()')
            keys = sorted(o, key=_SafeSortKey)
            return self._fmt_seq(dest, indent, ctx, keys, '{', '}')
        if isinstance(o, dict):
            return self._fmt_dict(dest, indent, ctx, o)
        if isinstance(o, frozenset):
            if len(o) == 0:
                return dest.write('frozenset()')
            keys = sorted(o, key=_SafeSortKey)
            return self._fmt_seq(dest, indent, ctx, keys, 'frozenset({', '})')
        # is_dataclass returns True for the class itself so check for that
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return self._fmt_dataclass(dest, indent, ctx, o)
        return self._fmt_fallback(dest, o)

    # noinspection PyMethodMayBeStatic
    def _fmt_fallback(self, dest: Dest, o: object):
        # We will have to be fine with it all being on one line
        return dest.write(repr(o))

    # Any is used for `dcls` as type system is too basic to represent all dataclasses as one
    def _fmt_dataclass(self, dest: Dest, indent: int, ctx: ContextStack,
                       dcls: Any):
        # Dataclasses have a reliable key order so don't sort them here
        # (For now also add the keys, TODO later might have an option for it?)
        self._fmt_seq(dest, indent, ctx, [
            StringJoin(DontReprMe(f'{f.name}='), getattr(dcls, f.name))
            for f in dataclasses.fields(dcls)
            if f.repr  # TODO maybe could be slightly cleverer than this (e.g. `or .compare`)
        ], f'{type(dcls).__name__}(', ')')

    # noinspection PyMethodMayBeStatic
    def _fmt_circular(self, dest: Dest, _indent: int, ctx: ContextStack, idx: int):
        # We would be inserted at ctx.len() so difference/how many to go up is the difference
        n_up = ctx.len() - idx
        dest.write(f'<Circular: {n_up} up>')

    def _fmt_dict(self, dest: Dest, indent: int, ctx: ContextStack, d: dict):
        items = sorted(d.items(), key=_dict_kv_sort_key)
        self._fmt_seq(dest, indent, ctx, [
            StringJoin(k, DontReprMe(': '), v)
            for k, v in items
        ], '{', '}')

    def _fmt_seq(self, dest: Dest, indent: int, ctx: ContextStack,
                 items: Sequence[object], start: str, end: str,
                 trailing_comma_if_unitary=False):
        # (Don't write indent on first line - handled by parent)
        if d := self._try_fmt_seq_short(ctx, items, start, end, trailing_comma_if_unitary):
            return dest.extend(d)
        dest.write(start, '\n')
        for i, v in enumerate(items):
            self._write_indent(dest, indent + 1)
            self._fmt(dest, indent + 1, ctx, v)
            if i != len(items) - 1:
                # No space after ',' if it's trailing
                dest.write(',\n')
        if trailing_comma_if_unitary and len(items) == 1:
            dest.write(',')
        dest.write('\n')
        self._indented_write(dest, indent, end)

    def _is_leaf(self, tp: type):
        for sup in tp.mro():  # MRO small => faster than iterating over leaf_types
            if sup in self.leaf_types:
                return True
        return False

    def _try_fmt_seq_short(self, ctx: ContextStack, items: Sequence[object],
                           start: str, end: str,
                           trailing_comma_if_unitary=False):
        if not all(self._is_leaf(type(o)) for o in items):
            return None  # Fast-track objects with non-leaf children
        dest = Dest()
        dest.write(start)
        # ^ Unlikely that `start` is already over the limit so don't check if we go over here
        for i, v in enumerate(items):
            self._fmt(dest, 0, ctx, v)
            if i != len(items) - 1:
                dest.write(', ')
            if dest.length() > self.max_simple_len:
                return None
        if trailing_comma_if_unitary and len(items) == 1:
            dest.write(',')
        dest.write(end)
        if dest.length() > self.max_simple_len:
            return None
        return dest

    def _write_indent(self, dest: Dest, indent: int):
        dest.write(indent * self.indent * ' ')

    def _indented_write(self, dest: Dest, indent: int, s: str):
        self._write_indent(dest, indent)
        dest.write(s)


def _dict_kv_sort_key(pair: tuple[object, object]):
    # Sort by key, then by value as fallback (in case of same str)
    return _SafeSortKey(pair[0]), _SafeSortKey(pair[1])


class _SafeSortKey:
    __slots__ = ['o']

    def __init__(self, o: object):
        self.o = o

    def _fallback_key(self) -> str:
        return str(self.o)

    def __lt__(self, other: _SafeSortKey):
        try:
            return self.o < other.o
        except TypeError:
            return self._fallback_key() < other._fallback_key()


class StringJoin:
    def __init__(self, *parts: DontReprMe | object):
        self.parts = parts


@dataclass
class DontReprMe:
    string: str


class ContextStack:
    def __init__(self):
        self._stack: list[int] = []  # list of ids
        self._ids: dict[int, int] = {}  # id->idx in stack

    @contextlib.contextmanager
    def enter_context(self, o: object):
        self.enter(o)
        try:
            yield
        finally:
            self.exit()

    def enter(self, o: object):
        assert self.get_idx(o) is None
        i = id(o)
        self._ids[i] = len(self._stack)
        self._stack.append(i)

    def exit(self):
        del self._ids[self._stack.pop()]  # Pop top id from stack and del it from the map

    def get_idx(self, o: object):
        return self._ids.get(id(o), None)

    def len(self):
        return len(self._stack)


class Dest:
    def __init__(self, *parts: str):
        self._parts = []
        self._len = 0
        self.write(*parts)

    def write(self, *args: str):
        for s in args:
            self._parts.append(s)
            self._len += len(s)

    def extend(self, other: Dest):
        self._parts.extend(other._parts)
        self._len += other._len

    def get_value(self):
        value = ''.join(self._parts)
        # Make later get_value()s faster at the cost of preventing changes to existing parts
        self._parts = [value]
        return value

    def length(self):
        return self._len
