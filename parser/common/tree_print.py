from __future__ import annotations

import sys
from io import StringIO
from typing import IO, Sequence

from ..cst.base_node import Leaf, AnyNode, Node
from .str_region import StrRegion

__all__ = [
    'TreePrinter', 'tree_print', 'tree_format', 'tprint', 'tformat'
]


class TreePrinter:
    def __init__(self, stream: IO[str] = None, indent=2, verbose=False, append_lf=False):
        self.indent = indent
        self.verbose = verbose
        self.append_lf = append_lf
        if stream is None:
            stream = sys.stdout
        self.stream = stream

    def print(self, obj: object):
        self._write(obj, 0)
        if self.append_lf:
            self.stream.write('\n')

    def format(self, obj: object):
        orig_stream = self.stream
        self.stream = out = StringIO()
        try:
            self.print(obj)
        finally:
            self.stream = orig_stream
        return out.getvalue()

    def _write(self, obj: object, level: int):
        if isinstance(obj, Node):
            self._write_node(obj, level)
        elif isinstance(obj, Leaf):
            self._write_leaf(obj, level)
        elif isinstance(obj, list):
            self._write_seq(obj, level, '[', ']')
        elif isinstance(obj, tuple):
            self._write_seq(obj, level, '(', ')', True)
        else:
            self._write_fallback(obj, level)

    def _write_seq(self, obj: Sequence, level: int, start: str,
                   end: str, require_trailing=False):
        self._indented_write(level, start)
        if len(obj) == 0:
            self.stream.write(end)
            return
        self.stream.write('\n')  # Start items on new line
        for i, c in enumerate(obj):
            self._write(c, level + 1)
            if i != len(obj) - 1:
                self.stream.write(',\n')
            elif require_trailing:
                self.stream.write(',')
        self.stream.write('\n')  # Put ']' on new line
        self._indented_write(level, end)

    def _write_node(self, obj: Node, level: int):
        start = self._fmt_node_header(obj, has_more_args=True) + '['
        self._write_seq(obj.children, level, start, end='])')

    def _write_leaf(self, obj: Leaf, level: int):
        self._indented_write(level, self._fmt_node_header(obj, has_more_args=False) + ')')

    def _write_fallback(self, obj: object, level: int):
        self._indented_write(level, repr(obj))

    def _indented_write(self, level: int, s: str):
        if level:
            self.stream.write(' ' * level * self.indent)
        self.stream.write(s)

    def _fmt_node_header(self, obj: AnyNode, has_more_args=True):
        args: list[str] = []
        if type(obj) == Leaf or type(obj) == Node:
            args.append(repr(obj.name))
        if self.verbose:
            args.append(self._fmt_region(obj.region))
        if has_more_args:
            args.append('')  # Placeholder for next arg so ', ' gets added
        return f'{type(obj).__name__}({", ".join(args)}'

    @classmethod
    def _fmt_region(cls, r: StrRegion):
        return f'StrRegion({r.start}, {r.end})'


def tree_print(obj: object, stream: IO[str] = None, indent: int = 2,
               verbose: bool = False, append_lf: bool = True):
    TreePrinter(stream, indent, verbose, append_lf).print(obj)


def tree_format(obj: object, indent: int = 2,
                verbose: bool = False, append_lf: bool = False):
    return TreePrinter(None, indent, verbose, append_lf).format(obj)


tprint = tree_print
tformat = tree_format
