from __future__ import annotations

from parser.common import BaseParseError, BaseLocatedError


class AstParseError(BaseParseError):
    pass


class LocatedAstError(BaseLocatedError, AstParseError):
    pass
