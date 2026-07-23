from __future__ import annotations

import dataclasses
import dataclasses as dcls
import enum
import sys
from collections import UserString
from io import StringIO
from typing import IO, Sequence

from util import dcls_field_default
from ..astgen.ast_node import AstNode
from ..cst.base_node import Leaf, AnyNode, Node
from .str_region import StrRegion

__all__ = [
    'TreePrinter', 'tree_print', 'tree_format', 'tprint', 'tformat'
]


@dataclasses.dataclass
class Concat:
    args: tuple[object, ...]

    def __init__(self, *args: object):
        self.args = args


class ConcatLiteral(UserString):
    pass


@dataclasses.dataclass
class IndentInfo:
    """Represents a first/current-line indentation and a possibly-different
    indentation for the later lines."""

    level: int
    start_level: int = dataclasses.field(default=-1)

    def __post_init__(self):
        if self.start_level == -1:
            self.start_level = self.level

    def subindent(self, n=1, indent_first_line=True):
        return type(self)(self.level + n, -1 if indent_first_line else 0)

    def get_indent(self, first_line: bool = False):
        return self.start_level if first_line else self.level


class TreePrinter:
    def __init__(self, stream: IO[str] | None = None, indent: int = 2,
                 verbose: bool = False, append_lf: bool = False):
        self.indent = indent
        self.verbose = verbose
        self.append_lf = append_lf
        if stream is None:
            stream = sys.stdout
        self.stream = stream

    def print(self, obj: object):
        self._write(obj, IndentInfo(0))
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

    def _write(self, obj: object, indent: IndentInfo):
        # TODO: general dataclass pretty-printing
        if isinstance(obj, Concat):
            self._write_concat(obj, indent)
        elif isinstance(obj, StrRegion):
            self._write_region(obj, indent)
        elif isinstance(obj, Node):
            self._write_cst_node(obj, indent)
        elif isinstance(obj, Leaf):
            self._write_cst_leaf(obj, indent)
        elif isinstance(obj, AstNode):
            self._write_ast_node(obj, indent)
        elif isinstance(obj, list):
            self._write_seq(obj, indent, '[', ']')
        elif isinstance(obj, tuple):
            self._write_seq(obj, indent, '(', ')', True)
        elif isinstance(obj, enum.Enum):
            self._write_enum(obj, indent)
        else:
            self._write_fallback(obj, indent)

    def _write_region(self, obj: StrRegion, indent: IndentInfo):
        self._indented_write(self._fmt_region(obj), indent, first_line=True)

    def _write_enum(self, obj: enum.Enum, indent: IndentInfo):
        self._indented_write(f'{type(obj).__name__}.{obj.name}', indent, first_line=True)

    def _write_concat(self, obj: Concat, indent: IndentInfo):
        self._indented_write('', indent, first_line=True)  # write start indent
        for o in obj.args:
            if isinstance(o, ConcatLiteral):
                self._indented_write(str(o), 0)
            else:
                self._write(o, indent.subindent(0, indent_first_line=False))

    def _write_seq(self, obj: Sequence[object], indent: IndentInfo, start: str,
                   end: str, require_trailing=False):
        self._indented_write(start, indent, first_line=True)
        if len(obj) == 0:
            self.stream.write(end)
            return
        self.stream.write('\n')  # Start items on new line
        for i, c in enumerate(obj):
            self._write(c, indent.subindent())
            if i != len(obj) - 1:
                self.stream.write(',\n')
            elif require_trailing and len(obj) == 1:
                self.stream.write(',')
        self.stream.write('\n')  # Put ']' on new line
        self._indented_write(end, indent)

    def _write_cst_node(self, obj: Node, indent: IndentInfo):
        start = self._fmt_node_header(obj, has_more_args=True) + '['
        self._write_seq(obj.children, indent, start, end='])')

    def _write_cst_leaf(self, obj: Leaf, indent: IndentInfo):
        self._indented_write(self._fmt_node_header(obj, has_more_args=False) + ')', indent)

    def _write_ast_node(self, obj: AstNode, indent: IndentInfo):
        # TODO Not the ideal formatting (but I don't know what is.
        field_values = [(f, getattr(obj, f.name)) for f in dcls.fields(obj)]
        pos_field_values = [(f, v) for f, v in field_values if not f.kw_only]
        while (pos_field_values and
               dcls_field_default(pos_field_values[-1][0])
               == pos_field_values[-1][1]):
            pos_field_values.pop(-1)  # start from last, remove default ones
        pos_values = [(None, v) for _f, v in pos_field_values]
        kw_values = [(f.name, v) for f, v in field_values
                     if f.kw_only and v != dcls_field_default(f)]
        values = pos_values + kw_values
        # Put the named ones at end. NOTE: relies on sort stability
        values.sort(key=lambda pair: pair[0] is not None)
        if ('meta', None) in values:  # special case: remove meta= if no metadata
            values.remove(('meta', None))
        assert values[0] == (None, obj.region)  # sanity check (region is first arg)
        args = [
            v if k is None else Concat(ConcatLiteral(f'{k}='), v)
            for k, v in values
        ]
        if self._is_complex(*args):
            return self._write_complex_ast_node(obj, args, indent)
        return self._write_simple_ast_node(obj, args, indent)

    @classmethod
    def _is_complex(cls, *args: object) -> bool:
        if len(args) != 1:
            return any(cls._is_complex(a) for a in args)
        (a,) = args
        if isinstance(a, Concat):
            return cls._is_complex(*a.args)
        return isinstance(a, (AnyNode, AstNode, list, tuple))

    def _write_simple_ast_node(self, obj: AstNode, args: list[object],
                               indent: IndentInfo):
        self._indented_write(f'{type(obj).__name__}(', indent, first_line=True)
        for i, v in enumerate(args):
            # The subindent is just in case one of them does contain a
            self._write(v, indent.subindent(indent_first_line=False))
            if i != len(args) - 1:
                self.stream.write(', ')
        self.stream.write(')')

    def _write_complex_ast_node(self, obj: AstNode, args: list[object],
                                indent: IndentInfo):
        start = f'{type(obj).__name__}({self._fmt_region(obj.region)},'
        del args[0]  # Remove the region argument
        return self._write_seq(args, indent, start, end=')')

    def _fmt_node_header(self, obj: AnyNode, has_more_args=True):
        args: list[str] = []
        if type(obj) == Leaf or type(obj) == Node:
            args.append(repr(obj.name))
        if self.verbose:
            args.append(self._fmt_region(obj.region))
        if has_more_args:
            args.append('')  # Placeholder for next arg so ', ' gets added
        return f'{type(obj).__name__}({", ".join(args)}'

    def _write_fallback(self, obj: object, level: IndentInfo):
        lines = repr(obj).splitlines()
        for i, ln in enumerate(lines):
            self._indented_write(ln, level, first_line=(i == 0))

    def _indented_write(self, s: str, level: IndentInfo | int, first_line: bool = False):
        if isinstance(level, IndentInfo):
            level = level.get_indent(first_line=first_line)
        if level:
            self.stream.write(' ' * level * self.indent)
        self.stream.write(s)

    @classmethod
    def _fmt_region(cls, r: StrRegion):
        return f'StrRegion({r.start}, {r.end})'


def tree_print(obj: object, stream: IO[str] | None = None, indent: int = 2,
               verbose: bool = False, append_lf: bool = True):
    TreePrinter(stream, indent, verbose, append_lf).print(obj)


def tree_format(obj: object, indent: int = 2,
                verbose: bool = False, append_lf: bool = False):
    return TreePrinter(None, indent, verbose, append_lf).format(obj)


tprint = tree_print
tformat = tree_format
