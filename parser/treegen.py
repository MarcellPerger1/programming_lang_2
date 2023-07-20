from __future__ import annotations

import bisect
import itertools
import traceback
from dataclasses import dataclass, field
from typing import (TypeVar, Iterable, cast, TypeAlias, Sequence, overload,
                    Callable, TYPE_CHECKING)

from parser.operators import UNARY_OPS, OPS_SET, COMPARISONS, ASSIGN_OPS
from parser.str_region import StrRegion
from parser.tokenizer import Tokenizer
from parser.tokens import *
from parser.tree_node import Node, Leaf, AnyNode
from parser.tree_print import tformat
from parser.treegen_matcher import OpM, KwdM, Matcher

if TYPE_CHECKING:
    from parser.treegen_matcher import PatternT

DT = TypeVar('DT')
ET = TypeVar('ET')

MISSING = object()

KEYWORDS = ['def', 'if', 'else', 'while', 'repeat', 'global', 'let']


class ParseError(SyntaxError):
    pass


class LocatedParseError(ParseError):
    def __init__(self, msg: str, region: StrRegion, src: str):
        super().__init__(msg)
        self.msg = msg
        self._src_text = src
        self.region = region

    def compute_location(self):
        # noinspection PyBroadException
        try:
            self.add_note(self.display_region(self._src_text, self.region))
        except Exception:
            short_loc = f'{self.region.start} to {self.region.end - 1}'
            self.add_note(
                '\nAn error occurred while trying to display the location '
                f'of the ParseError ({short_loc}):\n{traceback.format_exc()}')

    def __str__(self):
        self.compute_location()
        return super().__str__()

    @classmethod
    def display_region(cls, src: str, region: StrRegion) -> str:
        lines = src.splitlines(keepends=True)
        lengths = tuple(map(len, lines))
        cum_lengths = tuple(itertools.accumulate(lengths))
        start_idx = region.start
        end_idx = region.end - 1  # its inclusive here
        start_line, start_col = cls.idx_to_coord(cum_lengths, start_idx)
        end_line, end_col = cls.idx_to_coord(cum_lengths, end_idx)
        if start_line == end_line:
            return cls._display_single_line(lines, start_line, start_col, end_col)
        assert start_line < end_line
        return cls._display_multi_line(lines, start_line, start_col, end_line, end_col)

    @classmethod
    def _display_multi_line(cls, lines: Sequence[str], start_line: int, start_col: int,
                            end_line: int, end_col: int):
        if end_line - start_line + 1 > 5:
            # just print start and end lines
            lineno_w = len(str(end_line + 1))  # last will always be biggest
            start_repr = cls._display_single_line(
                lines, start_line, start_col, len(lines[start_line]) - 1, lineno_w)
            end_repr = cls._display_single_line(
                lines, end_line, 0, end_col, lineno_w)
            return (f'{start_repr}\n'
                    f'...\n'
                    f'{end_repr}')
        lines_repr = []
        lineno_w = len(str(end_line + 1))
        for line in range(start_line, end_line + 1):
            if line == start_line:
                start_col_inner = start_col
            else:
                start_col_inner = 0
            if line == end_line:
                end_col_inner = end_col
            else:
                end_col_inner = len(lines[line]) - 1
            line_repr = cls._display_single_line(
                lines, line, start_col_inner, end_col_inner, lineno_w)
            lines_repr.append(f'{line_repr}')
        return '\n'.join(lines_repr)

    @classmethod
    def _display_single_line(cls, lines: Sequence[str], line: int, start_col: int,
                             end_col: int, lineno_w: int = None):
        start_spaces = ' ' * start_col
        carets = '^' * (end_col - start_col + 1)
        end_spaces = ' ' * (len(lines[line]) - end_col)
        if lineno_w is None:
            lineno_w = len(str(line + 1))
        (line_str,) = lines[line].splitlines()  # remove \n on end
        return (f'{line + 1:>{lineno_w}} |  {line_str}\n'
                f'{""      :>{lineno_w}} |  {start_spaces}{carets}{end_spaces}')

    @classmethod
    def idx_to_coord(cls, cum_lengths: Sequence[int], idx: int) -> tuple[int, int]:
        """Converts an index to a **0-based** (line, column) tuple"""
        line = bisect.bisect_right(cum_lengths, idx)
        if line == 0:
            return line, idx
        # -1 to convert to idx, +1 to for char after last char on prev line
        line_start_idx = cum_lengths[line - 1]  # - 1 + 1
        col = idx - line_start_idx
        return line, col

    def add_note(self, param: str):
        # noinspection PyUnresolvedReferences
        super().add_note(param)


class TreeGen:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.src = self.tokenizer.src
        self.result: Node | None = None

    @property
    def all_tokens(self):
        return self.tokenizer.tokens

    @property
    def tokens(self):
        return self.tokenizer.content_tokens

    @overload
    def __getitem__(self, item: int) -> Token:
        ...

    @overload
    def __getitem__(self, item: slice) -> list[Token]:
        ...

    def __getitem__(self, item: int | slice):
        return self.tokens[item]

    def get(self, item: int | slice, default: DT = MISSING) -> Token | DT:
        try:
            return self[item]
        except IndexError:
            if default is MISSING:
                return self[-1]  # use pre-existing EofToken
            return default

    def eof(self, item: int):
        return item >= len(self.tokens)

    def matches(self, start: int, pattern: PatternT,
                default: bool = None, want_full=False):
        if default is None or 0 <= start < len(self.tokens):
            return self.match(start, pattern, want_full).success
        return default

    def match(self, start: int, pattern: PatternT, want_full=False):
        return Matcher(pattern, self.tokens, start, self.src).match(want_full)

    def parse(self):
        if not self.tokenizer.is_done:
            self.tokenizer.tokenize()
        assert isinstance(self.tokens[-1], EofToken)
        idx = 0
        smts = []
        while not self.eof(idx) and not self.matches(idx, EofToken):
            # todo what if eof in a _parse_* function?
            smt, idx = self._parse_smt(idx)
            smts.append(smt)
        node = Node('program', self.tok_region(0, idx), None, smts)
        self.result = node
        return node

    def _parse_smt(self, idx: int) -> tuple[AnyNode, int]:
        if self.matches(idx, (KwdM('def'), IdentNameToken, LParToken)):
            # Matches up to `def func(` We need to look this far because
            # 'def func = 2' is not a function ('def' is a type here)
            # but a variable declaration
            smt, idx = self._parse_define(idx)
        elif self.matches(idx, KwdM('if')):
            # 'if' is a keyword to make this matches() simpler
            # and so that potentially-infinite lookahead is not required
            # e.g. if(...) could be a function call or could be a part of
            # if(...) {} and '...' could be arbitrarily long
            smt, idx = self._parse_if(idx)
        elif self.matches(idx, KwdM('while')):
            smt, idx = self._parse_while(idx)
        elif self.matches(idx, KwdM('repeat')):
            smt, idx = self._parse_repeat(idx)
        elif self.matches(idx, (KwdM('global'), IdentNameToken)):
            smt, idx = self._parse_global(idx)
        elif self.matches(idx, (KwdM('let'), IdentNameToken)):
            smt, idx = self._parse_let(idx)
        elif self.matches(idx, IdentNameToken):
            smt, idx = self._parse_ident_at_start(idx)
        else:
            # can only be an expr
            # todo for now we are assuming that a smt can contain any expr
            #  but this could/will change in the future
            #  although it may be better not to deal with it here
            #  and instead do a post-processing step
            smt, idx, brk_reason = self._parse_expr(idx)
            if not self.matches(idx, SemicolonToken):
                raise self.err("Statements must end with a semicolon",
                               self[idx - 1]) from brk_reason
            idx += 1
        return smt, idx

    def _parse_ident_at_start(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        maybe_lvalue, idx = self._parse_maybe_lvalue(idx)

        next_token = self.get(idx)
        if isinstance(next_token, OpToken) and next_token.op_str in ASSIGN_OPS:
            # next is an assign so it *is* an lvalue
            return self._parse_lvalue_assign(maybe_lvalue, start, idx)
        # it's just an expr here
        expr, idx, brk_reason = self._parse_expr(start, maybe_lvalue, idx)
        if not self.matches(idx, SemicolonToken):
            raise ParseError("Expected ';' at end of expr") from brk_reason
        idx += 1
        return expr, idx

    def _parse_lvalue_assign(self, lvalue: AnyNode, smt_start: int,
                             assign_start: int) -> tuple[AnyNode, int]:
        idx = assign_start
        assign_token = self.get(idx)
        assert isinstance(assign_token, OpToken)
        assert assign_token.op_str in ASSIGN_OPS
        idx += 1
        expr, idx, brk_reason = self._parse_expr(idx)
        if not self.matches(idx, SemicolonToken):
            raise ParseError("Expected ';' at end of expr in assign") from brk_reason
        idx += 1
        return Node(assign_token.op_str, self.tok_region(smt_start, idx),
                    None, [lvalue, expr]), idx

    def _parse_maybe_lvalue(self, start: int) -> tuple[AnyNode, int]:
        """Parse tokens that are valid as an lvalue eg:
        - abc
        - x.y
        - ls[2]
        - w.0.x[1].abc
        """
        return self._parse_chained_gets(start)

    def _parse_chained_gets(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, IdentNameToken)
        curr = Leaf.of(self[idx])
        idx += 1
        return self._parse_get_chain_contd(idx, curr)

    def _parse_get_chain_contd(self, start: int, curr: AnyNode) -> tuple[AnyNode, int]:
        idx = start
        while True:
            if self.matches(idx, DotToken):
                idx += 1
                if not self.matches(idx, AttrNameToken):
                    raise ParseError("Expected attr_name after '.', "
                                     f"got {self[idx].name}")
                attr = Leaf.of(self[idx])
                idx += 1
                curr = Node('getattr', StrRegion(curr.region.start, attr.region.end),
                            None, [curr, attr])
            elif self.matches(idx, LSqBracket):
                sqb, idx = self._parse_sqb(idx)
                curr = Node('getitem', StrRegion(curr.region.start, sqb.region.end),
                            None, [curr, sqb])
            else:
                return curr, idx

    def _parse_sqb(self, start: int) -> tuple[AnyNode, int]:
        """Just returns the expr inside, idx=after end of sqb"""
        idx = start
        assert self.matches(idx, LSqBracket)
        idx += 1
        if self.matches(idx, RSqBracket):
            raise ParseError("Square brackets must not be empty")
        expr, idx, brk_reason = self._parse_expr(idx)
        if not self.matches(idx, RSqBracket):
            raise ParseError(
                f"Expected lsqb, got {self[idx].name}. (missing lsqb "
                f"or incomplete expression in square brackets)") from brk_reason
        idx += 1
        return expr, idx

    def _parse_let(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, KwdM('let'))
        idx += 1
        items, idx = self._parse_decl_item_list(idx)
        if not self.matches(idx, SemicolonToken):
            raise ParseError(f"Expected ';' or ',' after decl_item,"
                             f" got {self[idx].name}")
        idx += 1
        return Node('let_decl', self.tok_region(start, idx),
                    None, items), idx

    def _parse_decl_item_list(self, idx):
        items = []
        while True:
            glob, idx = self._parse_decl_item(idx)
            items.append(glob)
            if not self.matches(idx, CommaToken):
                break
            idx += 1
        return items, idx

    def _parse_global(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, KwdM('global'))
        idx += 1
        items, idx = self._parse_decl_item_list(idx)
        if not self.matches(idx, SemicolonToken):
            raise ParseError(f"Expected ';' or ',' after decl_item,"
                             f" got {self[idx].name}")
        idx += 1
        return Node('global_decl', self.tok_region(start, idx),
                    None, items), idx

    def _parse_decl_item(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        if not self.matches(idx, IdentNameToken):
            raise ParseError(f"Expected identifier in decl_item, "
                             f"got {self[idx].name}")
        ident = Leaf.of(self[idx])
        idx += 1
        # 1. global x, y = ...;
        #            ^
        # 2. global a=..., b;
        #            ^
        value: AnyNode | None = None
        if self.matches(idx, OpM('=')):
            # case 2
            idx += 1
            value, idx, brk_reason = self._parse_expr(idx)
            if not isinstance(self.get(idx), (SemicolonToken, CommaToken)):
                raise ParseError(f"Expected ';' or ',' after decl_item,"
                                 f" got {self[idx].name}") from brk_reason
        glob = Node('decl_item', self.tok_region(start, idx),
                    None, [ident, value])
        return glob, idx

    def _parse_define(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, KwdM('def'))
        idx += 1
        assert self.matches(idx, IdentNameToken)
        name = Leaf.of(self.tokens[idx])
        idx += 1
        args_decl, idx = self._parse_args_decl(idx)
        # def f(t1 arg1, t2 arg2) { <a block> }
        #                         ^
        block, idx = self._parse_block(idx)
        return Node('define', self.tok_region(start, idx),
                    None, [name, args_decl, block]), idx

    def _parse_args_decl(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        if not self.matches(idx, LParToken):
            raise ParseError("Expected '(' after 'def name'")
        idx += 1
        if self.matches(idx, RParToken):
            # simple case, no args
            idx += 1
            return Node('args_decl', self.tok_region(start, idx)), idx
        arg1, idx = self._parse_arg_decl(idx)
        arg_declares = [arg1]
        while not self.matches(idx, RParToken):
            if not self.matches(idx, CommaToken):
                raise ParseError(f"Expected ',' or ')' after arg_decl, got {self[idx].name}")
            idx += 1
            if self.matches(idx, RParToken):
                # def f(t1 arg1, t2 arg2,)
                #                       ~^
                break
            # def f(t1 arg1, t2 arg2)
            #              ~~^^
            arg, idx = self._parse_arg_decl(idx)
            arg_declares.append(arg)
        # def f(t1, t2)
        #             ^
        assert self.matches(idx, RParToken)
        idx += 1
        args_decl = Node('args_decl', self.tok_region(start, idx), None, arg_declares)
        return args_decl, idx

    def _parse_arg_decl(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        if not self.matches(idx, IdentNameToken):
            raise ParseError(f"Error: expected type name, got {self[idx].name}."
                             f"Did you forget a ')'?")
        tp_name = Leaf.of(self[idx])
        idx += 1
        if not self.matches(idx, IdentNameToken):
            raise ParseError(f"Error: expected arg name, got {self[idx].name}."
                             f"Did you forget the type name?")
        arg_name = Leaf.of(self[idx])
        idx += 1
        arg_decl = Node('arg_decl', self.tok_region(start, idx), None, [tp_name, arg_name])
        return arg_decl, idx

    def tok_region(self, start: int, end: int) -> StrRegion:
        start = self.get(start).region.start
        # end is exclusive
        end = self.get(end - 1).region.end
        return StrRegion(start, end)

    def _parse_block(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        if not self.matches(idx, LBrace):
            raise ParseError(f"Expected '{{' to start block, got {self[idx]}")
        idx += 1
        smts: list[AnyNode] = []
        while not self.matches(idx, RBrace):
            smt, idx = self._parse_smt(idx)
            smts.append(smt)
        if not self.matches(idx, RBrace):
            raise ParseError("Expected '}' to close block, got ")
        idx += 1
        return Node('block', self.tok_region(start, idx), None, smts), idx

    def _parse_while(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, KwdM('while'))
        idx += 1
        cond, idx, brk_reason = self._parse_expr(idx)
        if not self.matches(idx, LBrace):
            raise ParseError("Expected '{' after expr in while") from brk_reason
        block, idx = self._parse_block(idx)
        return Node('while', self.tok_region(start, idx),
                    None, [cond, block]), idx

    def _parse_repeat(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, KwdM('repeat'))
        idx += 1
        amount, idx, brk_reason = self._parse_expr(idx)
        if not self.matches(idx, LBrace):
            raise ParseError("Expected '{' after expr in repeat") from brk_reason
        block, idx = self._parse_block(idx)
        return Node('repeat', self.tok_region(start, idx),
                    None, [amount, block]), idx

    def _parse_if(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        if_part, idx = self._parse_if_cond(idx)
        elseif_parts = []
        else_part: AnyNode | None = None
        while self.matches(idx, KwdM('else')):
            if self.matches(idx + 1, KwdM('if')):
                elseif_part, idx = self._parse_elseif(idx)
                elseif_parts.append(elseif_part)
            elif self.matches(idx + 1, LBrace):
                else_part, idx = self._parse_else(idx)
                break
            else:
                raise ParseError(f"Expected '{{' or 'if' after 'else', "
                                 f"got {self[idx + 1].name}")
        if else_part is None:
            else_part = Node('else_cond_NULL', self.tok_region(idx - 1, idx - 1))
        return Node('if', self.tok_region(start, idx),
                    None, [if_part, *elseif_parts, else_part]), idx

    def _parse_if_cond(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, KwdM('if'))
        idx += 1
        cond, idx, brk_reason = self._parse_expr(idx)
        if not self.matches(idx, LBrace):
            raise ParseError("Expected '{' after expr in if") from brk_reason
        block, idx = self._parse_block(idx)
        return Node('if_cond', self.tok_region(start, idx),
                    None, [cond, block]), idx

    def _parse_elseif(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, (KwdM('else'), KwdM('if')))
        idx += 2
        cond, idx, brk_reason = self._parse_expr(idx)
        if not self.matches(idx, LBrace):
            raise ParseError("Expected '{' after expr in else if") from brk_reason
        block, idx = self._parse_block(idx)
        return Node('elseif_cond', self.tok_region(start, idx),
                    None, [cond, block]), idx

    def _parse_else(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, (KwdM('else'), LBrace))
        idx += 1  # don't advance past '{'; it's needed for _parse_block
        block, idx = self._parse_block(idx)
        return Node('else_cond', self.tok_region(start, idx), None, [block]), idx

    def _parse_call(self, left: AnyNode | Token, call_start: int) -> tuple[AnyNode, int]:
        if isinstance(left, Token):
            left = Leaf.of(left)
        idx = call_start
        args, idx = self._parse_call_args(idx)
        return Node('call', StrRegion(left.region.start, self[idx - 1].region.end),
                    None, [left, args]), idx

    def _parse_call_args(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, LParToken)
        idx += 1
        if self.matches(idx, RParToken):
            # simple case, no args
            idx += 1
            return Node('call_args', self.tok_region(start, idx)), idx
        arg1, idx, brk_reason = self._parse_expr(idx)
        args = [arg1]
        while not self.matches(idx, RParToken):
            if not self.matches(idx, CommaToken):
                raise ParseError(f"Expected ',' or ')' after arg, "
                                 f"got {self[idx].name}") from brk_reason
            idx += 1
            if self.matches(idx, RParToken):
                # f(arg1, arg2,)
                #             ~^
                break
            # f(arg1, arg2)
            #       ~~^
            arg, idx, brk_reason = self._parse_expr(idx)
            args.append(arg)
        # f(t1, t2)
        #         ^
        assert self.matches(idx, RParToken)
        idx += 1
        call_args = Node('call_args', self.tok_region(start, idx), None, args)
        return call_args, idx

    def _parse_grouping_parens(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, LParToken)
        idx += 1
        expr, idx, brk_reason = self._parse_expr(idx)
        if not self.matches(idx, RParToken):
            raise ParseError("Expected ')' at end of expr") from brk_reason
        idx += 1
        return Node('paren', self.tok_region(start, idx), None, [expr]), idx

    def _parse_expr(self, start: int, partial: AnyNode = None,
                    partial_end: int = None) -> tuple[AnyNode, int, ParseError | None]:
        curr, end, brk_reason = self._parse_expr_pass_0(start, partial, partial_end)
        curr = self._parse_expr_binary_op_level(
            curr, ('**', '**<unary>'),
            self._chained_pow_to_node, call_if_single=True)
        assert all(not isinstance(t, PseudoOpToken) for t in curr)
        curr = self._parse_expr_unary_ops(curr)
        curr = self._parse_ltr_op_level(curr, '*/')
        curr = self._parse_ltr_op_level(curr, '+-')
        curr = self._parse_ltr_op_level(curr, ['..'])
        curr = self._parse_expr_binary_op_level(
            curr, COMPARISONS, self._handle_chained_comp, call_if_single=False)
        curr = self._parse_ltr_op_level(curr, ['&&'])
        curr = self._parse_ltr_op_level(curr, ['||'])
        if len(curr) == 0:
            raise self.err(f"Expected expr, got no expr", self[start])
        if len(curr) != 1:
            raise self.err(
                f"Invalid expr (expected 1 node to be produced, "
                f"got {len(curr)} nodes). Debug: nodes={tformat(curr)}",
                curr
            ) from brk_reason
        (node,) = curr
        return node, end, brk_reason

    def _handle_chained_comp(
            self, chain: list[AnyNode | OpToken]
    ) -> AnyNode:  # type: ignore
        assert len(chain) >= 5
        assert len(chain) % 2 == 1
        raise self.err("Can't chain comparison operators", chain[3])

    def _parse_ltr_op_level(self, tokens: list[AnyNode | OpToken],
                            ops: str | Iterable[str]):
        return self._parse_expr_binary_op_level(tokens, ops, self._chained_ltr_grouping)

    def _chained_ltr_grouping(self, chain: list[AnyNode | OpToken]) -> AnyNode:
        assert len(chain) >= 3
        assert len(chain) % 2 == 1
        # functools.reduce() would be perfect for this
        left = chain[0]
        if isinstance(left, OpToken):
            raise self.err(f"Unexpected {left.op_str} at start of expr", left)
        for op_idx in range(1, len(chain), 2):
            op = chain[op_idx]
            assert isinstance(op, OpToken)
            right = chain[op_idx + 1]
            if isinstance(right, OpToken):
                raise self.err(f"Unexpected {right.op_str} after {op.op_str}", right)
            left = Node(op.op_str, left.region | right.region, None, [left, right])
        return left

    def _parse_expr_unary_ops(self, tokens: list[AnyNode | OpToken]):
        new = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if not isinstance(token, OpToken):
                new.append(token)
                idx += 1
                continue
            if not self._op_is_unary(tokens, idx):
                new.append(token)
                idx += 1
                continue
            if idx + 1 >= len(tokens):
                # last item
                raise self.err(f"Unexpected end of expr after {token.op_str}", token)
            arg = tokens[idx + 1]
            node = Node(token.op_str, token.region | arg.region, None, [arg])
            new.append(node)
            idx = (idx + 1) + 1
        return new

    # noinspection PyMethodMayBeStatic
    def _op_is_unary(self, tokens: list[Token | AnyNode], idx: int):
        assert idx >= 0
        token = tokens[idx]
        assert isinstance(token, OpToken)
        if token.op_str not in UNARY_OPS:
            return False
        if idx == 0 or token.op_str == '!':
            # definitely unary at start and '!' isn't valid as binary op
            return True
        if isinstance(prev_symbol := tokens[idx - 1], Token):
            return isinstance(prev_symbol, (OpToken, NullToken))
        assert isinstance(prev_symbol, AnyNode)
        return prev_symbol.name in OPS_SET

    # the reason for the complication here:
    # https://docs.python.org/3/reference/expressions.html#id21
    #  2**-1 => 2**(-1) NOT (2**-)1 meaning that
    #  unary +/- is more binding IF on rhs of **
    #  but is only bound to the pow and nothing else i.e.
    #  2**-3**4 => 2**(-(3**4))
    #          NOT 2**((-3)**4)
    #          NOT (2**-)(3**4)
    # Also
    #  1**--3 => 1**(-(-3))
    #        NOT (1'**-'-)3
    #  so this also needs to be called with len(chained) == 3
    # noinspection PyMethodMayBeStatic
    def _chained_pow_to_node(self, chained: list[AnyNode | OpToken]) -> AnyNode:
        assert len(chained) >= 3  # expr <op> expr
        assert len(chained) % 2 == 1
        curr = chained[-1]
        if isinstance(curr, OpToken):
            raise self.err(f"Unexpected {curr.op_str} after **", curr)
        # start at 2nd last elem (last op)
        for op_idx in range(len(chained) - 2, -1, -2):
            op = chained[op_idx]
            assert isinstance(op, OpToken)
            pre_arg = chained[op_idx - 1]
            assert isinstance(pre_arg, AnyNode)
            if op.op_str != '**':
                assert op.op_str == '**<unary>'
                assert isinstance(op, PseudoOpToken)
                assert len(op.inner_ops) >= 2
                unaries: list[OpToken]
                pow_token, *unaries = op.inner_ops
                for u in unaries[::-1]:
                    curr = Node(u.op_str, u.region | curr.region, None, [curr])
                op = pow_token
            assert op.op_str == '**'
            curr = Node('**', pre_arg.region | curr.region, None, [pre_arg, curr])
        return curr

    def _parse_expr_binary_op_level(
            self, tokens: list[OpToken | AnyNode], ops: str | Iterable[str],
            handle_chained: Callable[[list[AnyNode | OpToken]], AnyNode],
            call_if_single: bool = False
    ) -> list[AnyNode]:
        """
        handle_chained([node1, op1, node2, op2, node3, ...]) -> node

        node*: AnyNode | OpToken, op*: str"""
        if isinstance(ops, str):
            ops = tuple(ops)
        assert len(set(ops)) == len(ops)

        op_indices: list[tuple[int, OpToken]] = []
        for op in ops:
            op_indices += [
                (i, v) for i, v in enumerate(tokens)
                if isinstance(v, OpToken) and v.op_str == op
            ]
        op_indices.sort()

        grouped_indices = self._group_chained_ops(op_indices)

        new = []
        next_idx = 0
        op: OpToken
        item: tuple[int, OpToken] | list[tuple[int, OpToken]]
        for item in grouped_indices:
            if isinstance(item, tuple):
                # optimization for single operator in chain as it
                # doesn't need to sort out grouping order (except for pow)
                item: tuple[int, OpToken]
                idx, op = item
                # token directly before operator will be bound to operator instead
                new += tokens[next_idx: idx - 1]
                arg_0 = tokens[idx - 1]
                try:
                    arg_1 = tokens[idx + 1]
                except IndexError:
                    raise self.err(f"Unexpected end of expression after {op.op_str}", op)
                assert isinstance(arg_0, AnyNode)
                assert isinstance(arg_1, AnyNode)
                if call_if_single:
                    node = handle_chained([arg_0, op, arg_1])
                else:
                    node = Node(op.op_str, StrRegion.including(arg_0.region, arg_1.region),
                                None, [arg_0, arg_1])
                next_idx = (idx + 1) + 1
            else:
                # chained operators e.g. 3+4-2+9
                assert isinstance(item, list)
                first_idx = item[0][0]
                last_idx = item[-1][0]
                # token directly before operator will be bound to operator not token stream
                new += tokens[next_idx: first_idx - 1]
                op_seq: list[OpToken | AnyNode] = []
                for idx, op in item:
                    op_seq += [tokens[idx - 1], op]
                try:
                    op_seq.append(tokens[last_idx + 1])
                except IndexError:
                    last_token = item[-1][1]
                    raise self.err(f"Unexpected end of expression after"
                                   f" {last_token.op_str}", last_token)
                node = handle_chained(op_seq)
                next_idx = (last_idx + 1) + 1
            new.append(node)
        # add remaining tokens
        new += tokens[next_idx:]
        return new

    def _group_chained_ops(self, op_indices: list[tuple[int, OpToken]]):
        grouped_indices = []
        for ii, item in enumerate(op_indices):
            idx, op = item
            try:
                prev1_tuple = op_indices[ii - 1]
            except IndexError:
                grouped_indices.append(item)
                continue
            prev1, prev1_op = prev1_tuple

            if idx == prev1 + 1:
                raise self.err(f"Unexpected {op.op_str!r} after {prev1_op.op_str!r}", op)
            if idx == prev1 + 2:
                if isinstance(grouped_indices[-1], tuple):
                    # tuple *not list* means its single value
                    grouped_indices[-1] = [grouped_indices[-1], item]
                else:
                    grouped_indices[-1].append(item)
            else:
                grouped_indices.append(item)
        return grouped_indices

    def _parse_expr_pass_0(
            self, start: int, partial: AnyNode, partial_end: int
    ) -> tuple[TokensPass0_T, int, ParseError | None]:
        """Parses precedence levels 1-3 and determines end of expr

        Parses precedence levels 1-3:
         1. Grouping ( )
         2. String literal auto-concatenation ("xy" "_abc" => "xy_abc")
         3. Calling ( ), getitem [ ], getattr .
         4. 1. groups `**-` so that the pow is less tightly bound to the minus
            and also accepts another token after it
            so that 2**-3 => 2**(-3) NOT (2**-)3
            but 6**-7**8 => 6**(-(7**8)) NOT 6**((-7)**8)
        Also determines end of expr."""
        idx = start

        def lookbehind(target_idx: int, default: Token = None):
            if default is None:
                default = NullToken(self.tok_region(target_idx, target_idx))
            if target_idx < start:
                return default
            return self.get(target_idx)

        tokens: TokensPass0_T = []
        if partial is not None:
            tokens.append(partial)
            idx = partial_end
        brk_reason = None
        # parse basic chained_gets and parens and determine end of chain
        # NOTE: need to be careful with raising an error vs breaking
        # as valid parses could be produces by braking 1 token earlier
        # so only raise error if its definitely not a valid parse
        while True:
            if self.matches(idx, IdentNameToken):
                if (tok_str := self[idx].region.resolve(self.src)) in KEYWORDS:
                    # e.g. `for x of y`: `of` is keyword so stop paring expr
                    # and could still be a valid parse
                    brk_reason = ParseError(
                        f"Unexpected keyword {tok_str!r} "
                        f"in expr after {self.get(idx - 1).name}")
                    break
                if isinstance(lookbehind(idx - 1), IdentNameToken):
                    # e.g. `for x of y`: while paring 'x of y' @ `of`:
                    #  `of` is ident/soft keyword so
                    #  stop paring expr and could still be a valid parse
                    brk_reason = ParseError(
                        f"Expected operator-oid after ident, got ident")
                    break
                complex_ident, idx = self._parse_chained_gets(idx)
                tokens.append(complex_ident)
            elif self.matches(idx, DotToken):
                if lookbehind(idx - 1) not in GETATTR_VALID_AFTER_CLS:
                    # we just raise an error - there is no way this
                    # is a valid parse eg `[.abc]` is a syntax error
                    raise ParseError(f"Unexpected '.' after {self.get(idx - 1).name}")
                tokens[-1], idx = self._parse_get_chain_contd(idx, tokens[-1])
            elif self.matches(idx, LSqBracket):
                if lookbehind(idx - 1) not in GETATTR_VALID_AFTER_CLS:
                    raise ParseError(f"Unexpected '[' after {self.get(idx - 1).name}")
                tokens[-1], idx = self._parse_get_chain_contd(idx, tokens[-1])
            elif self.matches(idx, NumberToken):
                if not isinstance(lookbehind(idx - 1), NUMBER_VALID_AFTER):
                    # not valid for now but could be later
                    raise ParseError(f"Unexpected number after {self.get(idx - 1).name}")
                tokens.append(Leaf.of(self[idx]))
                idx += 1
            elif self.matches(idx, StringToken):
                prev_sym = _seq_get(tokens, -1, NullToken())
                if isinstance(prev_sym, StringToken):
                    # string literal auto-concatenation
                    region = StrRegion(prev_sym.region.start, self[idx].region.end)
                    tokens[-1] = Node('auto_concat', region,
                                      None, [prev_sym, Leaf.of(self[idx])])
                elif isinstance(prev_sym, Node) and prev_sym.name == 'auto_concat':
                    # string literal auto-concatenation 2
                    prev_sym.add(Leaf.of(self[idx]), update_end=True)
                elif isinstance(prev_sym, (OpToken, NullToken)):
                    tokens.append(Leaf.of(self[idx]))
                else:
                    raise ParseError(f"Unexpected string after {self.get(idx - 1).name}")
                idx += 1
            elif self.matches(idx, LParToken):
                if isinstance(lookbehind(idx - 1), LPAR_CALL_AFTER):
                    # fn call
                    # fn_call has same precedence as everything else handled
                    # here so it is fine to just use prev token here
                    # as more binding stuff have already been processed
                    tokens[-1], idx = self._parse_call(tokens[-1], idx)
                elif isinstance(lookbehind(idx - 1), LPAR_GROUPING_AFTER):
                    # grouping paren
                    inner, idx = self._parse_grouping_parens(idx)
                    tokens.append(inner)
                else:
                    # probably doesn't produce a valid parse as '(' is
                    # only used for grouping expressions
                    raise ParseError(f"Unexpected '(' after {self.get(idx - 1).name}")
            elif self.matches(idx, AttrNameToken):
                assert 0, "AttrNameToken should follow and be consumed by DotToken"
            elif self.matches(idx, OpToken):
                token = cast(OpToken, self[idx])
                if token.op_str == '=':
                    raise ParseError("Unexpected '=' in expression ("
                                     "assignment is only valid in statements)")
                if token.op_str == '**':
                    token, idx = self._handle_pow_pass_0(idx, token)
                else:
                    idx += 1
                tokens.append(token)
            elif isinstance(self.get(idx), EXPR_TERMINAL_TOKENS):
                break
            else:
                assert 0, ("This should be unreachable. _parse_expr reached"
                           f" unknown token {self.get(idx)}. "
                           f"Perhaps _parse_expr is outdated or broken? "
                           "Perhaps the tokenizer is broken or "
                           ".tokens is being used instead of .content_tokens?")
        end = idx
        return tokens, end, brk_reason

    def _handle_pow_pass_0(self, idx: int, token: OpToken) -> tuple[AnyNode | OpToken, int]:
        assert self[idx] == token
        assert token.op_str == '**'
        if not _is_unary_token(self.get(idx + 1)):
            return token, idx + 1
        tokens = [token]
        idx += 1
        while _is_unary_token(next_t := self.get(idx)):
            tokens.append(next_t)
            idx += 1

        res = PseudoOpToken(
            StrRegion.including(*[t.region for t in tokens]),
            '**<unary>',
            tokens)
        return res, idx

    def err(self, msg: str, loc: (
            Token | AnyNode | Sequence[Token] | Sequence[AnyNode] | StrRegion)):
        if not isinstance(loc, StrRegion):
            try:
                seq: tuple[Token | AnyNode, ...] = tuple(loc)
            except TypeError:
                seq = (loc,)
            regs: list[StrRegion] = []
            for o in seq:
                regs.append(o.region)
            region = StrRegion.including(*regs)
        else:
            region = loc
        return LocatedParseError(msg, region, self.src)


def _is_unary_token(t: Token):
    return isinstance(t, OpToken) and t.op_str in UNARY_OPS


@dataclass
class PseudoOpToken(OpToken):
    inner_ops: list[OpToken] = field(default_factory=list)


TokensPass0_T: TypeAlias = 'list[AnyNode | OpToken]'

EXPR_BEGIN = (
    CommaToken,
    SemicolonToken,
    LParToken,
    LSqBracket,
    LBrace,
    RBrace,
    NullToken
)
UNARY_VALID_AFTER = (
    OpToken,
    *EXPR_BEGIN
)


# concat (..) precedence?!: least binding:
#  "a" .. 8 - 6 .. "b"  =>  "a" .. (8 - 6) .. "b"

# operator precedence (most to least binding):
#  1. () [] {}  (parens)
#  2. string literal auto-concatenation ("ab" "xy" => "abxy")
#  3. . [] fn_call() (getattr, getitem, function call)
#  4.
#  4.1 **-
#  4.2 ** (bare)
#  5. + - ! (unary)
#  6. * /
#  7. + -
#  8. ..
#  9. == != < > <= >=
# 10. &&
# 11. ||


EXPR_TERMINAL_TOKENS = (
    EofToken,
    CommaToken,  # fn_args(<expr>, ...)
    SemicolonToken,  # smt: <expr>;
    RParToken,  # in parens: 9*(<expr>)*...
    RSqBracket,  # in computed getitem: list[<expr>]
    LBrace,  # in block start: if <expr>{...}
    # not sure if RBrace is terminal and not just an error:
    # if true{<expr>} is an error (no ';' after expr)
    # (need ';' at end of smt but...)
    # It's probably best to say that it is terminal and let the caller
    # (e.g. _parse_smt) deal with it as it can be more elegantly handled there,
    # perhaps with better and more specialized error messages.
    RBrace,
)
NUMBER_VALID_AFTER = (
    NullToken,
    OpToken
)
# NOTE: use isinstance() with these NOT `in`
LSQB_VALID_AFTER = (
    AnyNameToken,
    NumberToken,
    StringToken,
    RParToken,
    RSqBracket
)
LPAR_CALL_AFTER = (
    RParToken,
    AnyNameToken,
    RSqBracket
)
LPAR_GROUPING_AFTER = (
    CommaToken,
    SemicolonToken,
    OpToken,
    LParToken,
    LSqBracket,
    LBrace,
    RBrace,
    NullToken
)


INVALID_AFTER_IDENT = (
    IdentNameToken,
    NumberToken,
    StringToken
)


def _seq_get(self: Sequence[ET], item: int, default: DT) -> ET | DT:
    try:
        return self[item]
    except IndexError:
        return default
