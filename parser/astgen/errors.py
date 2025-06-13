from __future__ import annotations

from ..common import BaseParseError, BaseLocatedError


class AstParseError(BaseParseError):
    pass


class LocatedAstError(BaseLocatedError, AstParseError):
    pass


class AstStringParseError(LocatedAstError):
    pass
