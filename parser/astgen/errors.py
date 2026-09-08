from __future__ import annotations

from ..common import BaseLocatedError


class AstParseError(BaseLocatedError):
    pass


class AstStringParseError(AstParseError):
    pass
