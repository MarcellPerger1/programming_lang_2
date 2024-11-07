from __future__ import annotations

import unicodedata

from .errors import LocatedAstError
from ..common import StrRegion


class AstStringParseError(LocatedAstError):
    ...


def eval_number(src: str):
    # All allowed numbers should satisfy float()'s requirements
    try:
        return float(src)
    except ValueError as e:
        raise AssertionError(
            "There is a bug in tokenizer's _NumberParser. AST received a "
            "number node that Python can't parser") from e


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
_HEX_ESCAPES = {
    # Where h/H = a hex digit (hexit???)
    'x': 2,  # \xHH  (2 extra chars)
    'u': 4,  # \uHHHH
    'U': 8,  # \Uhhhhhhhh
}

HEX_DIGITS = set('0123456789abcdef')


def eval_string(s: str, region: StrRegion, full_src: str):
    s = s[1:-1]  # Exclude first and last char (start/end quotes)
    region = StrRegion(region.start + 1, region.end - 1)
    chars = []  # Build it into a list to prevent O(n^2) concat on non-CPython
    i = 0
    while i < len(s) - 1:
        start_i = i
        c = s[i]
        i += 1
        if c != '\\':
            chars.append(c)
            continue
        c = s[i]
        i += 1
        try:
            chars.append(_SINGLE_CHAR_ESCAPES[c])
            continue
        except KeyError:
            pass
        if length := _HEX_ESCAPES.get(c):
            digits = s[i:i+length].lower()
            try:
                value = int(digits, base=16)
            except ValueError:
                # Just report entire escape as error region
                raise AstStringParseError(
                    f"Invalid escape in string (expected {length} hex digits)",
                    StrRegion(region.start + start_i,
                              region.start + i + length),
                    full_src) from None
            chars.append(chr(value))
            i = i + length
            continue
        # Very special case: \N{......}
        if c != 'N':
            raise AstStringParseError(
                f"Unknown string escape \\{c}",
                StrRegion(region.start + start_i, region.start + i), full_src)
        (c,) = s[i:i+1] or '\0'
        if c != '{':
            raise AstStringParseError(
                "Expected '{' after \\N in string escape",
                StrRegion(region.start + start_i, region.start + i), full_src)
        i += 1
        end_idx = s.find('}', i)
        if end_idx == -1:
            raise AstStringParseError(
                "Expected unicode-name-escape to be terminated by a '}'",
                StrRegion(region.start + start_i, region.end), full_src)
        name = s[i:end_idx]
        if not name:
            raise AstStringParseError(
                f"Cannot have empty character name in string escape",
                StrRegion(region.start + start_i, region.start + end_idx + 1),
                full_src)
        try:
            chars.append(unicodedata.lookup(name))
            i = end_idx + 1
            continue
        except KeyError as e:
            raise AstStringParseError(
                f"Unknown unicode character name in escape: '{name}'",
                StrRegion(region.start + start_i, region.start + end_idx + 1),
                full_src) from e
    return ''.join(chars)
