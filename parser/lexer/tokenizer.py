from __future__ import annotations

import sys
from io import StringIO
from string import ascii_letters, digits
from typing import TYPE_CHECKING, IO, Sequence

from parser.error import BaseParseError, BaseLocatedError
from parser.operators import OPS_SET, MAX_OP_LEN, OP_FIRST_CHARS
from parser.str_region import StrRegion
from parser.tokens import (
    Token, WhitespaceToken, LineCommentToken, BlockCommentToken, NumberToken,
    StringToken, CommaToken, DotToken, OpToken, PAREN_TYPES,
    SemicolonToken, AttrNameToken, IdentNameToken,
    GETATTR_VALID_AFTER_CLS, EofToken
)
if TYPE_CHECKING:
    from parser.tokens import ParenTokenT

IDENT_START = ascii_letters + '_'
IDENT_CONT = IDENT_START + digits


class TokenizerError(BaseParseError):
    ...


class LocatedTokenizerError(BaseLocatedError, TokenizerError):
    ...


class MalformedNumberError(TokenizerError):
    ...


class LocatedMalformedNumberError(LocatedTokenizerError, MalformedNumberError):
    ...


class SrcHandler:
    def __init__(self, src: str):
        self.src: str = src

    def __getitem__(self, item: int | slice) -> str:
        return self.src[item]

    def eof(self, idx: int):
        return idx >= len(self.src)

    def get(self, idx: int, eof: str = '\0') -> str:
        try:
            return self.src[idx]
        except IndexError:
            return eof

    default_err_type = LocatedTokenizerError

    def err(self, msg: str,
            loc: int | Token | StrRegion | Sequence[int | Token | StrRegion],
            tp: type[BaseLocatedError] = None):
        try:
            seq: tuple[int | Token | StrRegion, ...] = tuple(loc)
        except TypeError:
            seq = (loc,)
        regs = []
        for o in seq:
            if isinstance(o, int):
                reg = StrRegion(o, o + 1)
            elif isinstance(o, Token):
                reg = o.region
            else:
                reg = o
            regs.append(reg)
        region = StrRegion.including(*regs)
        if tp is None:
            tp = self.default_err_type
        return tp(msg, region, self.src)


class Tokenizer(SrcHandler):
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
            # order fastest ones first
            if self[idx] == ',':
                idx = self._t_comma(idx)
            elif self[idx] == '.':
                # this is the complicated case where we need to decide:
                # is it a NumberToken, a GetattrToken or a OpToken('..')
                # need to use get as file could end on the '.'
                if self.get(idx + 1) == '.':
                    idx = self._t_concat(idx)
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
                # need to decide between number and attr_name here...
                # except if a number can take the pace of an attr name.
                # No, that wouldn't work:
                # abc.12.3e9 => abc.(12.3e9) which is bad, needs to be
                # abc.12.3e9 => (abc.12).3e9
                # BAD:  abc.91.7 => abc.(91.7)
                # GOOD: abc.91.7 => (abc.91).7
                # meaning if last 'real' token is a DOT,
                # this should be treated as an attr
                if self.prev_content_token_type is DotToken:
                    idx = self._t_attr_name(idx)
                else:
                    idx = self._t_number(idx)
            elif self[idx] in IDENT_START:
                # decide whether its attr_name or ident_name
                if self.prev_content_token_type is DotToken:
                    idx = self._t_attr_name(idx)
                else:
                    idx = self._t_ident_name(idx)
            elif self[idx] in PAREN_TYPES:
                tp: ParenTokenT = PAREN_TYPES[self[idx]]
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
        return self[start: start + len(s)].startswith(s)

    def _t_space(self, start: int):
        idx = start
        if not self[idx].isspace():
            return start
        while self.get(idx).isspace():
            idx += 1
        return self.add_token(
            WhitespaceToken(StrRegion(start, idx)), whitespace=True)

    def _t_line_comment(self, start: int) -> int:
        idx = start
        if not self.startswith(idx, '//'):
            return start
        idx += 2
        while self.get(idx) != '\n':
            idx += 1
        return self.add_token(LineCommentToken(StrRegion(start, idx)))

    def _t_block_comment(self, start: int) -> int:
        idx = start
        if not self.startswith(idx, '/*'):
            return start
        idx += 2
        while not self.startswith(idx, '*/'):
            idx += 1
        idx += 2  # include '*/' in comment
        return self.add_token(BlockCommentToken(StrRegion(start, idx)))

    def _t_comma(self, start: int) -> int:
        idx = start
        if self[idx] != ',':
            return start
        idx += 1
        return self.add_token(CommaToken(StrRegion(start, idx)))

    def _t_string(self, start: int) -> int:
        idx = start
        if self[idx] not in '\'"':
            return start
        q_type = self[idx]
        idx += 1
        while True:
            if self.eof(idx):
                raise self.err(f"Unexpected EOF in string "
                               f"(expected {q_type} to close string)", idx)
            if self[idx] == q_type:
                idx += 1
                return self.add_token(StringToken(StrRegion(start, idx)))
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
    def _t_number(self, start: int) -> int:
        # doesn't handle negative numbers,
        # those should be handled as a separate '-' operator
        idx = start
        assert self[idx] != '-', "_t_number doesn't handle negative numbers"
        tok, idx = _IncrementalNumberParser(self.src).parse(idx)
        if tok is not None:
            self.add_token(tok)
        return idx

    def _t_dot(self, start: int) -> int:
        idx = start
        assert self[idx] == '.', "_t_dot should only be called if char is '.'"
        idx += 1
        return self.add_token(DotToken(StrRegion(start, idx)))

    def _t_concat(self, start: int) -> int:
        idx = start
        assert self.startswith(idx, '..')
        idx += 2
        return self.add_token(OpToken(StrRegion(start, idx), '..'))

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


class _IncrementalNumberParser(SrcHandler):
    default_err_type = LocatedMalformedNumberError

    # todo 0x, 0b
    def _parse_digit_seq(self, start: int) -> int | None:
        idx = start
        if self.get(idx) not in digits:
            if self.get(idx) == '_':
                raise self.err("Can't have '_' at the start of a number", idx)
            return None
        idx += 1
        while True:
            if self.get(idx) == '_':
                if self.get(idx + 1) in digits:
                    idx += 2  # '_' and digit
                elif self.get(idx + 1) == '_':
                    raise self.err(
                        "Can only have one consecutive '_' in a number", idx + 1)
                else:
                    raise self.err(
                        "Can't have '_' at the end of a number", idx)
            elif self.get(idx) in digits:
                idx += 1
            else:
                return idx

    def _parse_num_no_exp(self, start: int) -> int | None:
        idx = start
        new_idx = self._parse_digit_seq(idx)
        if new_idx is not None:
            has_pre_dot = True
            idx = new_idx
        else:
            # eg: .234, e7, abcdef
            if self.get(idx) != '.':
                return None
            has_pre_dot = False
        if self.get(idx) != '.':
            # eg: 1234, 567e-5, 8 +9-10
            return idx
        idx += 1
        new_idx = self._parse_digit_seq(idx)
        if new_idx is not None:
            has_post_dot = True
            idx = new_idx
        else:
            has_post_dot = False
        if not has_pre_dot and not has_post_dot:
            # or maybe raise an error?
            # abc.e definitely doesn't contain a number
            # but it *could* reach here to check if it's a number
            #   \- Note from later: it doesn't.
            # return None
            raise self.err("Number cannot be a single '.' (expected digits before or after", idx)
        return idx

    def _parse_number(self, start: int):
        idx = start
        idx = self._parse_num_no_exp(idx)
        if idx is None:
            return None
        if self.get(idx).lower() != 'e':
            return idx
        idx += 1
        # need to handle '-' here explicitly as it is part of the number
        # so can't just be parsed as a separate operator
        if self.get(idx) == '-':
            idx += 1
        new_idx = self._parse_digit_seq(idx)  # no dot after the 'e'
        if not new_idx:
            # eg: 1.2eC, 8e-Q which is always an error
            raise self.err("Expected integer after <number>e", idx)
        idx = new_idx
        return idx

    def parse(self, start: int) -> tuple[Token | None, int]:
        idx = self._parse_number(start)
        if idx is None:
            return None, start
        return NumberToken(StrRegion(start, idx)), idx


def main():
    import pprint
    src = ("ab2 = 12==!(  \n" + r"e3.1.2 >= '7\'\\')%12;var x='b'..7")
    t = Tokenizer(src)
    t.tokenize()
    pprint.pp(t.tokens)
    table = []
    for tok in t.tokens:
        if tok.is_whitespace:
            row = ['(WS) ' + repr(tok.region.resolve(src)), tok.name]
        else:
            row = [str(tok.region.resolve(src)), tok.name]
        table.append(row)
    max0 = max(len(r[0]) for r in table)
    max1 = max(len(r[1]) for r in table)
    for s0, s1 in table:
        print(f'{s0:>{max0}} {s1:>{max1}}')


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


if __name__ == '__main__':
    main()
