from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..common import HasRegion, StrRegion

__all__ = [
    'AnyCommentToken', 'AnyNameToken', 'AttrNameToken', 'BlockCommentToken',
    'CommaToken', 'DotToken', 'EofToken',
    'IdentNameToken', 'LBrace', 'LParToken', 'LSqBracket', 'LineCommentToken',
    'NamedTokenCls', 'NumberToken', 'OpToken', 'PAREN_TYPES', 'ParenSide',
    'ParenToken', 'ParenType', 'RBrace', 'RParToken', 'RSqBracket',
    'SemicolonToken',  'StringToken', 'Token', 'WhitespaceToken',
]


@dataclass
class Token(HasRegion):
    name: str
    region: StrRegion = None
    # not a field but a class var:
    is_whitespace = False  # type: bool

    def isinst(self, cls: type[Token]):
        if type(self) == Token:
            # self is a general Token class so can't compare classes
            # so compare names but doesn't work for subclasses
            # Also check for isinstance for Token().isinst(Token) == True
            return isinstance(self, cls) or self.name == cls.name
        return isinstance(self, cls)

    def get_str(self, full_str: str):
        assert self.region is not None, "Token.get_str requires region to be set"
        return self.region.resolve(full_str)


@dataclass
class NamedTokenCls(Token):
    name: str = field(init=False, repr=False)

    def __post_init__(self):
        if type(self) is NamedTokenCls:
            raise TypeError("NamedTokenCls may not be instantiated directly;"
                            " use a subclass or use Token")


class WhitespaceToken(NamedTokenCls):
    name = 'whitespace'
    is_whitespace = True


class AnyCommentToken(NamedTokenCls):
    """Base class for comments"""
    is_whitespace = True


class LineCommentToken(AnyCommentToken):
    name = 'line_comment'


class BlockCommentToken(AnyCommentToken):
    name = 'block_comment'


class NumberToken(NamedTokenCls):
    name = 'number'


class StringToken(NamedTokenCls):
    name = 'string'


class CommaToken(NamedTokenCls):
    name = 'comma'


class DotToken(NamedTokenCls):
    name = 'dot'


@dataclass
class OpToken(NamedTokenCls):
    name = 'op'
    op_str: str = None


@dataclass
class ParenToken(NamedTokenCls):
    # base class for LPar, RPar
    side = None  # type: ParenSide
    paren_type = None  # type: ParenType
    paren_str = None  # type: str


def register_paren_cls(char: str):
    def inner(cls: type[ParenToken]):
        PAREN_TYPES[cls] = char
        PAREN_TYPES[char] = cls
        cls.paren_str = char
        return cls

    return inner


PAREN_TYPES: dict[str | type[ParenToken], type[ParenToken] | str] = {}


class ParenSide(Enum):
    LEFT = 0
    RIGHT = 1


class ParenType(Enum):
    NORMAL = 0
    SQUARE = 1
    CURLY = 2


@register_paren_cls('(')
class LParToken(ParenToken):
    name = 'lpar'
    side = ParenSide.LEFT
    paren_type = ParenType.NORMAL


@register_paren_cls(')')
class RParToken(ParenToken):
    name = 'rpar'
    side = ParenSide.RIGHT
    paren_type = ParenType.NORMAL


@register_paren_cls('[')
class LSqBracket(ParenToken):
    name = 'lsqb'
    side = ParenSide.LEFT
    paren_type = ParenType.SQUARE


@register_paren_cls(']')
class RSqBracket(ParenToken):
    name = 'rsqb'
    side = ParenSide.RIGHT
    paren_type = ParenType.SQUARE


@register_paren_cls('{')
class LBrace(ParenToken):
    name = 'lbrace'
    side = ParenSide.LEFT
    paren_type = ParenType.CURLY


@register_paren_cls('}')
class RBrace(ParenToken):
    name = 'rbrace'
    side = ParenSide.RIGHT
    paren_type = ParenType.CURLY


class SemicolonToken(NamedTokenCls):
    name = 'semi'


class AnyNameToken(NamedTokenCls):
    ...


class AttrNameToken(AnyNameToken):
    name = 'attr_name'


class IdentNameToken(AnyNameToken):
    name = 'ident_name'


class EofToken(NamedTokenCls):
    name = 'eof'
