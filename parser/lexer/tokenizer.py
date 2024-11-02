from __future__ import annotations

import sys
from io import StringIO
from string import ascii_letters, digits
from typing import IO

from .number_parser import NumberParser
from .src_handler import UsesSrc
from .tokens import *
from ..common import StrRegion
from ..operators import OPS_SET, MAX_OP_LEN, OP_FIRST_CHARS


IDENT_START = ascii_letters + '_'
IDENT_CONT = IDENT_START + digits


class Tokenizer(UsesSrc):
    def __init__(self, src: str):
        super().__init__(src)
        self.tokens: list[Token] = []
        self.content_tokens: list[Token] = []
        self.is_done = False

    @property
    def prev_content_token_type(self):
        if len(self.content_tokens) == 0:
            return None
        return type(self.content_tokens[-1])

    def tokenize(self):
        idx = 0
        last_idx = -1
        while idx < len(self.src):
            if idx == last_idx:
                raise RuntimeError(
                    f"No progress made after one tokenizer iteration at {idx=}."
                    f" This is a bug in the tokenizer.")
            else:
                last_idx = idx
            # order fastest comparisons/most common chars first
            if self[idx] == ',':
                idx = self.add_token(CommaToken(StrRegion(idx, idx + 1)))
            elif self[idx] == '.':
                # this is the complicated case where we need to decide:
                # is it a NumberToken, a GetattrToken or a OpToken('..')
                # need to use get as file could end on the '.'
                if self.get(idx + 1) == '.':
                    idx = self.add_token(OpToken(StrRegion(idx, idx + 2), '..'))
                elif self.prev_content_token_type in GETATTR_VALID_AFTER_CLS:
                    idx = self._t_dot(idx)
                else:
                    idx = self._t_number(idx)
            elif self[idx] == ';':
                idx = self.add_token(SemicolonToken(StrRegion(idx, idx + 1)))
            # these 2 take roughly equal time but
            # much more chars will just be whitespace than quotes
            elif self[idx].isspace():
                idx = self._t_space(idx)
            elif self[idx] in '\'"':
                idx = self._t_string(idx)
            elif self[idx] in digits:
                # Can only be an attribute if prev 'real' token is a dot.
                if self.prev_content_token_type is DotToken:
                    # If prev token is a dot, it MUST be an attribute as
                    # numbers aren't valid after '.'
                    idx = self._t_attr_name(idx)
                else:
                    # Otherwise, it can only be a number
                    idx = self._t_number(idx)
            elif self[idx] in IDENT_START:
                # decide whether its attr_name or ident_name
                if self.prev_content_token_type is DotToken:
                    idx = self._t_attr_name(idx)
                else:
                    idx = self._t_ident_name(idx)
            elif self[idx] in PAREN_TYPES:
                tp = PAREN_TYPES[self[idx]]
                idx = self.add_token(tp(StrRegion(idx, idx + 1)))
            elif self[idx:idx+2] == '//':
                idx = self._t_line_comment(idx)
            elif self[idx:idx+2] == '/*':
                idx = self._t_block_comment(idx)
            # this is one of the most expensive checks
            # so try to do the other ones first
            elif (self._could_be_op(idx)
                  and (new_idx := self._t_op(idx)) is not None):
                idx = new_idx
            else:
                raise self.err(f"No token matches source from {idx=}", idx)
        self.add_token(EofToken(StrRegion(idx, idx)))
        self.is_done = True
        return self

    def add_token(self, *tokens: Token, whitespace=None) -> int | None:
        if not tokens:
            return None
        self.tokens += tokens
        if whitespace is None:
            self.content_tokens += (t for t in tokens if not t.is_whitespace)
        elif not whitespace:
            self.content_tokens += tokens
        return tokens[-1].region.end

    def startswith(self, start: int, s: str):
        assert not self.eof(start), "You probably shouldn't try to startswith after EOF"
        return self.src.startswith(s, start)

    def _t_space(self, start: int):
        idx = start
        assert self[idx].isspace(), "_t_space should only be called on spaces"
        while self.get(idx).isspace():
            idx += 1
        return self.add_token(
            WhitespaceToken(StrRegion(start, idx)), whitespace=True)

    def _t_line_comment(self, start: int) -> int:
        idx = start
        assert self.startswith(idx, '//')
        idx += 2
        while not self.eof(idx) and self.get(idx) != '\n':
            idx += 1
        return self.add_token(LineCommentToken(StrRegion(start, idx)))

    def _t_block_comment(self, start: int) -> int:
        idx = start
        if not self.startswith(idx, '/*'):
            return start
        idx += 2
        while not self.eof(idx) and not self.startswith(idx, '*/'):
            idx += 1
        if self.eof(idx):
            raise self.err("Unterminated block comment", StrRegion(start, idx))
        idx += 2  # include '*/' in comment
        return self.add_token(BlockCommentToken(StrRegion(start, idx)))

    def _t_string(self, start: int) -> int:
        idx = start
        # Note: the assert and assignment cannot be merged as Python wouldn't
        # execute the assignment in `-O` optimisation mode
        assert self[idx] in '\'"'
        q_type = self[idx]
        idx += 1
        while True:
            if self.eof(idx):
                raise self.err(f"Unexpected EOF in string "
                               f"(expected {q_type} to close string)", idx)
            if self[idx] == q_type:
                idx += 1
                return self.add_token(StringToken(StrRegion(start, idx)))
            # TODO: maybe somehow attach the escapes to the Token
            #  so it doesn't need to be parsed again
            if self[idx] == '\\':
                # 1 for the '\', 1 for the next char
                idx += 2
            else:
                idx += 1

    # we may have a problem here:
    # how to tokenize abc.23? It could be:
    # 1. IDENT(abc) FLOAT(.23)
    # 2. IDENT(abc) GETATTR(.) ATTR_IDENT(23)
    # It should obviously result in 2.
    # but it may be tricky to implement. Although...
    # If we check for a valid getattr first based on the previous tokens
    # (yes, I know looking at previous tokens is not good),
    # and if there isn't one, try to do the float
    def _t_number(self, idx: int) -> int:
        # doesn't handle negative numbers,
        # those should be handled as a separate '-' operator
        assert self[idx] != '-', "_t_number doesn't handle negative numbers"
        return self.add_token(NumberParser(self.src).parse(idx))

    def _t_dot(self, idx: int) -> int:
        assert self[idx] == '.', "_t_dot should only be called if char is '.'"
        return self.add_token(DotToken(StrRegion(idx, idx + 1)))

    def _could_be_op(self, start: int) -> bool:
        return self[start] in OP_FIRST_CHARS

    def _t_op(self, start: int) -> int | None:
        idx = start
        # optimisation... only look at next `length`
        # chars so == can be used (-1 startswith() method call)
        # this also means that we can check for the first `length`
        # chars in the set, turning this into a O(number of unique lengths)
        # operation, instead of O(number of operators)
        for length in range(MAX_OP_LEN, 0, -1):  # check longest first
            next_op = self[idx: idx+length]
            if next_op in OPS_SET:
                idx += length
                return self.add_token(OpToken(StrRegion(start, idx), next_op))
        return None

    def _t_attr_name(self, start: int) -> int:
        idx = start
        assert self[idx] in IDENT_CONT
        while self.get(idx) in IDENT_CONT:
            idx += 1
        return self.add_token(AttrNameToken(StrRegion(start, idx)))

    def _t_ident_name(self, start: int) -> int:
        idx = start
        assert self[idx] in IDENT_START
        idx += 1
        while self.get(idx) in IDENT_CONT:
            idx += 1
        return self.add_token(IdentNameToken(StrRegion(start, idx)))


GETATTR_VALID_AFTER_CLS = (
    StringToken,
    RParToken,
    RSqBracket,
    AttrNameToken,
    IdentNameToken
    # Not valid (directly) after floats (need parens) because we treat all
    # numbers the same and we cannot have it after ints
    #   2.3 => (2).3 (attribute) or `2.3` (float)
    # Also it would be confusing to have 2.e3 => num, 2.e3.3 -> num.attr.
)


def print_tokens(src: str, tokens: list[Token], stream: IO[str] = None, do_ws=False):
    if stream is None:
        stream = sys.stdout
    table = []
    for tok in tokens:
        if tok.is_whitespace:
            if do_ws:
                table.append(['(WS) ' + repr(tok.region.resolve(src)), tok.name])
        else:
            table.append([str(tok.region.resolve(src)), tok.name])
    max0 = max(len(r[0]) for r in table)
    max1 = max(len(r[1]) for r in table)
    for s0, s1 in table:
        print(f'{s0:>{max0}} | {s1:>{max1}}', file=stream)


def format_tokens(src: str, tokens: list[Token], do_ws=False):
    out = StringIO()
    print_tokens(src, tokens, out, do_ws)
    return out.getvalue()
