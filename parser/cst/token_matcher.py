from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast, Iterable, Any, TypeGuard, TypeAlias

from parser.tokens import Token, OpToken, IdentNameToken


@dataclass
class MatchResult:
    matched: bool
    next_idx: int


class BaseMatcher(ABC):
    @abstractmethod
    def matches(self, tokens: list[Token], start: int, src: str) -> MatchResult:
        ...


class _FastTokenClsMatcher(BaseMatcher):
    def __init__(self, pattern: type[Token]):
        self.pattern = pattern

    def matches(self, tokens: list[Token], start: int, src: str) -> MatchResult:
        matched = isinstance(tokens[start], self.pattern)
        # could be written as `return MatchResult(matched, start + matched)`
        if matched:
            return MatchResult(True, start + 1)
        return MatchResult(False, start)


class TokenMatcher(BaseMatcher):
    def __init__(self, pattern: type[Token] | Token):
        self.pattern = pattern

    # in general, try to use subclass/instance check
    # as we also should allow subclasses to be matched e.g.
    # AnyParenToken matches LPar(...) => True
    def _is_cls_matched(self, token: Token) -> bool:
        assert inspect.isclass(self.pattern)
        if self.pattern == Token:
            return True
        elif type(token) == Token:
            # created using general Token ctor so just check name
            return self.pattern.name == token.name
        return isinstance(token, self.pattern)

    def _is_inst_matched(self, token: Token) -> bool:
        assert not inspect.isclass(self.pattern)
        if type(self.pattern) == Token or type(token) == Token:
            # one of them is not a concrete type so compare names
            return token.name == self.pattern.name
        return isinstance(token, type(self.pattern))

    def matches(self, tokens: list[Token], start: int, src: str) -> MatchResult:
        token = tokens[start]
        if inspect.isclass(self.pattern):
            matched = self._is_cls_matched(token)
        else:
            matched = self._is_inst_matched(token)
        if matched:
            return MatchResult(True, start + 1)
        return MatchResult(False, start)


class OpMatcher(BaseMatcher):
    def __init__(self, op_str: str):
        self.op = op_str

    def matches(self, tokens: list[Token], start: int, src: str) -> MatchResult:
        token = tokens[start]
        if token.isinst(OpToken) and self.op == cast(OpToken, token).op_str:
            return MatchResult(True, start + 1)
        return MatchResult(False, start)


class KwdMatcher(BaseMatcher):
    def __init__(self, kwd: str, base_type: type[Token] | None = None):
        self.kwd = kwd
        if base_type is None:
            base_type = IdentNameToken
        self.base_type = base_type

    def matches(self, tokens: list[Token], start: int, src: str) -> MatchResult:
        token = tokens[start]
        if token.isinst(self.base_type) and token.region.resolve(src) == self.kwd:
            return MatchResult(True, start + 1)
        return MatchResult(False, start)


class SeqMatcher(BaseMatcher):
    def __init__(self, *matchers: BaseMatcher):
        self.matchers = matchers

    def matches(self, tokens: list[Token], start: int, src: str) -> MatchResult:
        idx = start
        for m in self.matchers:
            m_result = match(m, tokens, idx, src)
            if not m_result.matched:
                return MatchResult(False, start)
            idx = m_result.next_idx
        return MatchResult(True, idx)


TokenM = TokenMatcher
OpM = OpMatcher
KwdM = KwdMatcher
SeqM = SeqMatcher


def _issubclass(cls: object, cls_or_tuple: type | tuple[type, ...]):
    try:
        return issubclass(cls, cls_or_tuple)  # type: ignore
    except TypeError:
        if inspect.isclass(cls):
            # cls *is* a class so __instancecheck__ raised a TypeError
            # and *not* isinstance so should probably let it propagate?
            raise
        return False


class Matcher:
    # start argument is to save memory as then the list slices
    # don't have to be stored separately
    def __init__(self, pattern: Iterable | type[Token] | Token | BaseMatcher,
                 tokens: list[Token], start: int, src: str):
        self.src = src
        self.pattern = pattern
        self.tokens = tokens
        self.start = start
        self.result: MatchResult | None = None

    def match(self, want_full=False):
        self._match()
        if want_full and self.next_idx != len(self.tokens):
            raise ValueError("Matcher dod not reach end")
        return self

    @property
    def success(self):
        return self.result.matched

    @property
    def next_idx(self):
        return self.result.next_idx

    def _match(self):
        if (isinstance(self.pattern, type)
                and issubclass(self.pattern, Token)
                and self.pattern != Token):
            # optimize the common case of `matches(..., <TokenCls>)`
            self.pattern = _FastTokenClsMatcher(self.pattern)
        elif isinstance(self.pattern, Token) or _issubclass(self.pattern, Token):
            self.pattern = TokenMatcher(self.pattern)
        elif isiterable(self.pattern):
            self.pattern = SeqMatcher(*cast(Iterable[BaseMatcher], self.pattern))
        assert isinstance(self.pattern, BaseMatcher)
        self.result = self.pattern.matches(self.tokens, self.start, self.src)


def matches(pattern: Iterable | type[Token] | Token | BaseMatcher,
            tokens: list[Token], start: int, src: str,
            want_complete: bool = False) -> bool:
    return Matcher(pattern, tokens, start, src).match(want_complete).success


def match(pattern: Iterable | type[Token] | Token | BaseMatcher,
          tokens: list[Token], start: int, src: str,
          want_complete: bool = False) -> MatchResult:
    return Matcher(pattern, tokens, start, src).match(want_complete).result


def isiterable(o: Any) -> TypeGuard[Iterable]:
    try:
        iter(o)
    except TypeError:
        return False
    return True


PatternT: TypeAlias = Iterable['PatternT'] | type[Token] | Token | BaseMatcher
