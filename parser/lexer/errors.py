from __future__ import annotations

from parser.common import BaseParseError, BaseLocatedError


class TokenizerError(BaseParseError):
    ...


class LocatedTokenizerError(BaseLocatedError, TokenizerError):
    ...


class MalformedNumberError(TokenizerError):
    ...


class LocatedMalformedNumberError(LocatedTokenizerError, MalformedNumberError):
    ...
