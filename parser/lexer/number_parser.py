from __future__ import annotations

from string import digits

from parser.common import StrRegion
from parser.lexer import LocatedMalformedNumberError
from parser.lexer.src_handler import UsesSrc
from parser.lexer.tokens import Token, NumberToken


class NumberParser(UsesSrc):
    default_err_type = LocatedMalformedNumberError

    # todo 0x, 0b (I refuse to add octal literals) - also hex floats???
    def _parse_digit_seq(self, start: int) -> int | None:
        # (Returns None if no digits)
        idx = start
        if self.get(idx) == '_':
            raise self.err("Can't have '_' at the start of a number", idx)
        if self.get(idx) not in digits:
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
                return idx  # end of digits/'_'

    def _parse_num_no_exp(self, idx: int) -> int:
        new_idx = self._parse_digit_seq(idx)
        if new_idx is None:
            if self.get(idx) != '.':
                raise self.err("Number must start with digit or '.' ", idx)
            has_pre_dot = False
        else:
            has_pre_dot = True
            idx = new_idx
        if self.get(idx) != '.':
            # eg: 1234, 567e-5, 8 +9-10
            return idx
        idx += 1
        new_idx = self._parse_digit_seq(idx)
        if new_idx is None:
            has_post_dot = False
        else:
            has_post_dot = True
            idx = new_idx
        if has_pre_dot or has_post_dot:
            return idx
        raise self.err("Number cannot be a single '.' "
                       "(expected digits before or after)", idx)

    def _parse_number(self, idx: int) -> int:
        idx = self._parse_num_no_exp(idx)
        if self.get(idx).lower() != 'e':
            return idx
        idx += 1
        # need to handle '-' here explicitly as it is part of the number
        # so can't just be parsed as a separate operator
        if self.get(idx) == '-':
            idx += 1
        new_idx = self._parse_digit_seq(idx)  # no dot after the 'e'
        if new_idx is None:
            # eg: 1.2eC, 8e-Q which is always an error
            raise self.err("Expected integer after <number>e", idx)
        idx = new_idx
        return idx

    def parse(self, start: int) -> Token:
        return NumberToken(StrRegion(start, self._parse_number(start)))
