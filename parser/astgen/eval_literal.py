from __future__ import annotations

import unicodedata

from .errors import LocatedAstError
from ..common import StrRegion


class AstStringParseError(LocatedAstError):
    ...


_MISSING = object()


def eval_number(src: str):
    # All allowed numbers should satisfy float()'s requirements
    try:
        return float(src)
    except ValueError as e:
        raise AssertionError(
            "There is a bug in tokenizer's _NumberParser. AST received a "
            "number node that Python can't parser") from e


def eval_string(s: str, region: StrRegion, full_src: str):
    return _EvalString(s, region, full_src).eval_string()


class _EvalString:
    def __init__(self, s: str, region: StrRegion, full_src: str):
        self.s = s[1:-1]  # Remove start/end quotes
        self.region = StrRegion(region.start + 1, region.end - 1)
        self.full_src = full_src

    def __getitem__(self, item):
        return self.s[item]

    def eval_string(self):
        chars = []  # Build it into a list to prevent O(n^2) concat on non-CPython
        i = 0
        while i < len(self.s):
            start_i = i
            c = self[i]
            i += 1
            if c != '\\':
                chars.append(c)  # The common case (no escape)
                continue
            c = self[i]
            i += 1
            if (res := _SINGLE_CHAR_ESCAPES.get(c)) is not None:
                res, i = res, i  # Don't need to consume any extra
            elif length := _HEX_ESCAPES.get(c):
                res, i = self._handle_hex_escape(length, i, start_i)
            elif c == 'N':
                res, i = self._handle_named_unicode_escape(i, start_i)
            else:
                raise self.err("Unknown string escape \\{c}", StrRegion(start_i, i))
            chars.append(res)
        return ''.join(chars)

    def _handle_hex_escape(self, length, i, start_i):
        digits = self[i:i + length].lower()
        i = i + length
        if len(digits) != length:  # Got less than expected, so end of string
            raise self.err(
                f"Unterminated escape in string (expected {length}"
                f" hex digits)", StrRegion(start_i, i))
        try:
            value = int(digits, base=16)
        except ValueError:
            # Just report entire escape as error region
            raise self.err(
                f"Invalid escape in string (expected {length} hex digits)",
                StrRegion(start_i, i)) from None
        return (chr(value)), i

    # the Very Special Case: \N{......}
    def _handle_named_unicode_escape(self, i: int, start_i: int):
        # This method takes over right after the `\N` bit (`{` is next)
        (c,) = self[i:i + 1] or '\0'  # This makes it end-of-string-proof
        if c != '{':
            raise self.err("Expected '{' after \\N in string escape",
                           StrRegion(start_i, i))
        i += 1
        end_idx = self.s.find('}', i)
        if end_idx == -1:
            raise self.err(
                "Expected unicode-name-escape to be terminated by a '}'",
                StrRegion(start_i, len(self.s)))
        name = self[i:end_idx]
        i = end_idx + 1  # Also consume the `}`
        if not name:
            raise self.err("Cannot have empty character name in string escape",
                           StrRegion(start_i, i))
        try:
            return unicodedata.lookup(name), i
        except KeyError as e:
            raise self.err(
                f"Unknown unicode character name in escape: '{name}'",
                StrRegion(start_i, i)) from e

    def err(self, msg: str, subregion: StrRegion):
        region = StrRegion(self.region.start + subregion.start,
                           self.region.start + subregion.end)
        if region.end > self.region.end:  # Clamp to within region
            region.end = self.region.end
        if region.start < self.region.start:
            region.start = self.region.start
        return AstStringParseError(msg, region, self.full_src)


_SINGLE_CHAR_ESCAPES = {
    '\n': '',       # \<newline> ignored
    '\\': '\\',     # \\ -> \
    "'": "'",       # \' -> '
    '"': '"',       # \" -> "
    'a': '\a',
    'b': '\b',
    'f': '\f',
    'n': '\n',
    'r': '\r',
    't': '\t',
    'v': '\v',
    '0': '\0',
}
# noinspection SpellCheckingInspection
_HEX_ESCAPES = {  # Gives number of extra chars
    # Where h/H = a hex digit (hexit???)
    'x': 2,  # \xHH  (2 extra chars)
    'u': 4,  # \uHHHH
    'U': 8,  # \Uhhhhhhhh
}
