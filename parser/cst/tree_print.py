from __future__ import annotations

import sys
from io import StringIO
from typing import IO, Sequence

from parser.tree_node import Node, Leaf

__all__ = [
    'TreePrinter', 'tree_print', 'tree_format', 'tprint', 'tformat'
]


class TreePrinter:
    def __init__(self, stream: IO[str] = None, indent=2):
        self.indent = indent
        if stream is None:
            stream = sys.stdout
        self.stream = stream

    def print(self, obj: object):
        self._write(obj, 0)
        self.stream.write('\n')

    def format(self, obj: object):
        out = StringIO()
        orig_stream = self.stream
        try:
            self.stream = out
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

    def _write_seq(self, obj: Sequence, level: int, char_start: str,
                   char_end: str, require_trailing=False):
        indent = ' ' * level * self.indent
        self.stream.write(f'{indent}{char_start}')
        if len(obj) == 0:
            self.stream.write(f'{char_end}')
            return
        else:
            self.stream.write('\n')
        for i, c in enumerate(obj):
            self._write(c, level + 1)
            if i != len(obj) - 1:
                self.stream.write(',\n')
            elif require_trailing:
                self.stream.write(',')
        self.stream.write(f'\n{indent}{char_end}')

    def _write_node(self, obj: Node, level: int):
        indent = ' ' * level * self.indent
        self.stream.write(f'{indent}Node({obj.name}, [')
        if len(obj.children) == 0:
            self.stream.write('])')
            return
        else:
            self.stream.write('\n')
        for i, c in enumerate(obj.children):
            self._write(c, level + 1)
            if i != len(obj.children) - 1:
                self.stream.write(',\n')
        self.stream.write(f'\n{indent}])')

    def _write_leaf(self, obj: Leaf, level: int):
        indent = ' ' * level * self.indent
        self.stream.write(f'{indent}Leaf({obj.name})')

    def _write_fallback(self, obj: object, level: int):
        indent = ' ' * level * self.indent
        self.stream.write(f'{indent}{obj!r}')


def tree_print(obj: object, stream: IO[str] = None, indent: int = 2):
    if stream is None:
        stream = sys.stdout
    TreePrinter(stream, indent).print(obj)


def tree_format(obj: object, indent: int = 2):
    out = StringIO()
    tree_print(obj, out, indent)
    return out.getvalue()


tprint = tree_print
tformat = tree_format
