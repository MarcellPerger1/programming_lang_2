from __future__ import annotations

from dataclasses import dataclass

import bisect
import itertools
from dataclasses import replace as d_replace
from typing import Sequence

__all__ = ['StrRegion']


@dataclass
class StrRegion:
    start: int
    end: int

    def resolve(self, s: str):
        return s[self.start:self.end]

    @classmethod
    def including(cls, *regions: StrRegion):
        """Returns smallest region that fits regions"""
        if len(regions) == 0:
            raise ValueError("Union of 0 regions")
        return StrRegion(min([r.start for r in regions]),
                         max([r.end for r in regions]))

    def union(self, *regions: StrRegion):
        return self.including(self, *regions)

    def __or__(self, other: StrRegion):
        return self.union(other)

    def is_epsilon(self):
        """Return true if the length of the region is 0"""
        return self.end <= self.start

    def display(self, src: str):
        # Logic held in separate class as it is rather complex
        return _RegionPrinter.display_region(src, self)


class _RegionPrinter:
    @classmethod
    def display_region(cls, src: str, region: StrRegion) -> str:
        if region.end > len(src):  # end excl so ok if 1 beyond string
            region = d_replace(region, end=len(src))
        if region.start >= len(src):
            region = d_replace(region, start=len(src) - 1)
        if region.is_epsilon():
            # length=0 but still try to do something
            region = d_replace(region, end=region.start + 1)
        lines = src.splitlines(keepends=True)
        cum_lengths = tuple(itertools.accumulate(map(len, lines)))
        start_idx = region.start
        end_idx = region.end - 1  # its inclusive below so convert excl -> incl
        start_line, start_col = cls._idx_to_coord(cum_lengths, start_idx)
        end_line, end_col = cls._idx_to_coord(cum_lengths, end_idx)
        if start_line == end_line:
            return cls._display_single_line(lines, start_line, start_col, end_col)
        assert start_line < end_line
        return cls._display_multi_line(lines, start_line, start_col, end_line, end_col)

    @classmethod
    def _display_multi_line(cls, lines: Sequence[str], start_line: int, start_col: int,
                            end_line: int, end_col: int):
        n_middle_lines = end_line - start_line - 1
        if n_middle_lines > 3:
            # just print start and end lines
            lineno_w = len(str(end_line + 1))  # last will always be biggest
            start_repr = cls._display_single_line(
                lines, start_line, start_col, len(lines[start_line]) - 1, lineno_w)
            end_repr = cls._display_single_line(
                lines, end_line, 0, end_col, lineno_w)
            return (f'{start_repr}\n'
                    f'... <{n_middle_lines} lines omitted>\n'
                    f'{end_repr}')
        lines_repr = []
        lineno_w = len(str(end_line + 1))
        for line in range(start_line, end_line + 1):
            if line == start_line:
                start_col_inner = start_col
            else:
                start_col_inner = 0
            if line == end_line:
                end_col_inner = end_col
            else:
                end_col_inner = len(lines[line]) - 2  # another for the \n at end
            line_repr = cls._display_single_line(
                lines, line, start_col_inner, end_col_inner, lineno_w)
            lines_repr.append(f'{line_repr}')
        return '\n'.join(lines_repr)

    @classmethod
    def _display_single_line(cls, lines: Sequence[str], line: int, start_col: int,
                             end_col: int, lineno_w: int | None = None):
        start_spaces = ' ' * start_col
        carets = '^' * (end_col - start_col + 1)
        end_spaces = ' ' * (len(lines[line]) - end_col)
        if lineno_w is None:
            lineno_w = len(str(line + 1))
        (line_str,) = lines[line].splitlines()  # remove \n on end
        return (f'{line + 1:>{lineno_w}} |  {line_str}\n'
                f'{""      :>{lineno_w}} |  {start_spaces}{carets}{end_spaces}')

    @classmethod
    def _idx_to_coord(cls, cum_lengths: Sequence[int], idx: int) -> tuple[int, int]:
        """Converts an index to a **0-based** (line, column) tuple"""
        line = bisect.bisect_right(cum_lengths, idx)
        if line == 0:
            return line, idx
        # -1 to convert to idx, +1 to for char after last char on prev line
        line_start_idx = cum_lengths[line - 1]  # - 1 + 1
        col = idx - line_start_idx
        return line, col
