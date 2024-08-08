from __future__ import annotations

import bisect
import itertools
import traceback
from dataclasses import replace as d_replace
from typing import Sequence

from parser.str_region import StrRegion


# We unconditionally add this polyfill to ensure that
# exceptions are always part of str(exc).
# Note that the 'really clever' way of implementing this
# (@property __notes__) FAILS because traceback also uses
# __notes__ to print the extra non-str() stuff on the end.
# So we cannot distinguish between whether it was called
# by add_note() (and we should return a ref to __polyfill_notes__)
# or by print_exc() (and we should return [] so notes are only added once)
# (Well except by looking at stack frames and similar hackery
#  but even I'm not willing to go that far!
#  (and it might fail when called from C extension))
# This means that the rest of the code will just have to deal with it
# and use get_notes() instead of .__notes__
class _PolyfillAddNoteMixin:
    """Usage: class MyError(_PolyfillAddNoteMixin, Exception): ...
    (**The order is important!**)"""
    # Don't use actual attribute because then we would include it in str()
    # and notes would also be printed below by traceback.print_exc (so twice)
    __polyfill_notes__: list[str]

    # We are first in the MRO so we override Exception's stuff for notes
    def add_note(self, note: str, /):
        self._transfer_builtin_notes()
        self._add_note(note)

    def get_notes(self):
        return getattr(self, '__polyfill_notes__', [])

    def _add_note(self, note: str, /):
        if not hasattr(self, '__polyfill_notes__'):
            self.__polyfill_notes__ = []
        self.__polyfill_notes__.append(note)

    # It is possible that we are actually not the first in MRO
    # (because of incorrect usage?) so we provide this method
    # to transfer the notes to our polyfill
    def _transfer_builtin_notes(self):
        for note in getattr(self, '__notes__', None) or []:
            self._add_note(str(note))
        try:
            # noinspection PyUnresolvedReferences
            del self.__notes__  # Clear added
        except AttributeError:
            pass

    def __str__(self):
        self._transfer_builtin_notes()
        # here super() refers to the next thing in MRO,
        # meaning the class after this in the bases
        if notes := getattr(self, '__polyfill_notes__', []):
            return super().__str__() + '\n' + '\n'.join(notes)
        return super().__str__()


class BaseParseError(_PolyfillAddNoteMixin, Exception):
    pass


class BaseLocatedError(BaseParseError):
    __notes__: list[str]  # but might not exist

    def __init__(self, msg: str, region: StrRegion, src: str):
        super().__init__(msg)
        self.msg = msg
        self._src_text = src
        self.region = region
        self._has_added_note = False

    # Don't need to send over traceback (for now) as Python prints
    # all the errors from other processes as well .
    # (But it would be a nice challenge for when I have time)
    # (Then again so would a lot of other projects I want to do)
    def __getnewargs__(self):
        # This function is needed here because pickle is a bit stupid
        # (dunno why?) so need to tell it the new args.
        return self.msg, self.region, self._src_text

    def compute_location(self):
        if self._has_added_note:
            return
        # noinspection PyBroadException
        try:
            self.add_note(self.display_region(self._src_text, self.region))
        except Exception:
            short_loc = f'{self.region.start} to {self.region.end - 1}'
            self.add_note(
                '\nAn error occurred while trying to display the location '
                f'of the ParseError ({short_loc}):\n{traceback.format_exc()}')
        self._has_added_note = True

    def __str__(self):
        self.compute_location()
        return super().__str__()

    @classmethod
    def display_region(cls, src: str, region: StrRegion) -> str:
        if region.end > len(src):  # end exc so ok if 1 beyond string
            region = d_replace(region, end=len(src))
        if region.start >= len(src):
            region = d_replace(region, start=len(src) - 1)
        if region.is_epsilon():
            # length=0 but still try to do something
            region = d_replace(region, end=region.start + 1)
        lines = src.splitlines(keepends=True)
        lengths = tuple(map(len, lines))
        cum_lengths = tuple(itertools.accumulate(lengths))
        start_idx = region.start
        end_idx = region.end - 1  # its inclusive here
        start_line, start_col = cls.idx_to_coord(cum_lengths, start_idx)
        end_line, end_col = cls.idx_to_coord(cum_lengths, end_idx)
        if start_line == end_line:
            return cls._display_single_line(lines, start_line, start_col, end_col)
        assert start_line < end_line
        return cls._display_multi_line(lines, start_line, start_col, end_line, end_col)

    @classmethod
    def _display_multi_line(cls, lines: Sequence[str], start_line: int, start_col: int,
                            end_line: int, end_col: int):
        if end_line - start_line + 1 > 5:
            # just print start and end lines
            lineno_w = len(str(end_line + 1))  # last will always be biggest
            start_repr = cls._display_single_line(
                lines, start_line, start_col, len(lines[start_line]) - 1, lineno_w)
            end_repr = cls._display_single_line(
                lines, end_line, 0, end_col, lineno_w)
            return (f'{start_repr}\n'
                    f'...\n'
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
                             end_col: int, lineno_w: int = None):
        start_spaces = ' ' * start_col
        carets = '^' * (end_col - start_col + 1)
        end_spaces = ' ' * (len(lines[line]) - end_col)
        if lineno_w is None:
            lineno_w = len(str(line + 1))
        (line_str,) = lines[line].splitlines()  # remove \n on end
        return (f'{line + 1:>{lineno_w}} |  {line_str}\n'
                f'{""      :>{lineno_w}} |  {start_spaces}{carets}{end_spaces}')

    @classmethod
    def idx_to_coord(cls, cum_lengths: Sequence[int], idx: int) -> tuple[int, int]:
        """Converts an index to a **0-based** (line, column) tuple"""
        line = bisect.bisect_right(cum_lengths, idx)
        if line == 0:
            return line, idx
        # -1 to convert to idx, +1 to for char after last char on prev line
        line_start_idx = cum_lengths[line - 1]  # - 1 + 1
        col = idx - line_start_idx
        return line, col
