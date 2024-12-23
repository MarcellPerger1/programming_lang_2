from __future__ import annotations

from typing import (TypeVar, cast, Sequence, overload, Iterable, Callable)

from .base_node import AnyNode, Node
from .named_node import AnyNamedNode, node_from_token, node_cls_from_name
from .nodes import *
from .token_matcher import OpM, KwdM, Matcher, PatternT
from ..common import StrRegion, region_union, RegionUnionArgT
from ..common.error import BaseParseError, BaseLocatedError
from ..lexer import Tokenizer
from ..operators import UNARY_OPS, COMPARISONS, ASSIGN_OPS
from ..tokens import *

DT = TypeVar('DT')

MISSING = object()

KEYWORDS = ['def', 'if', 'else', 'while', 'repeat', 'global', 'let']


class CstParseError(BaseParseError):
    pass


class LocatedCstError(BaseLocatedError, CstParseError):
    pass


class CstGen:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.src = self.tokenizer.src
        self.result: ProgramNode | None = None

    @property
    def all_tokens(self):
        return self.tokenizer.tokens

    @property
    def tokens(self):
        return self.tokenizer.content_tokens

    @overload
    def __getitem__(self, item: int) -> Token: ...
    @overload
    def __getitem__(self, item: slice) -> list[Token]: ...

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
        if self.result:
            return self.result
        if not self.tokenizer.is_done:
            self.tokenizer.tokenize()
        assert isinstance(self.tokens[-1], EofToken)
        idx = 0
        smts = []
        while not self.eof(idx) and not self.matches(idx, EofToken):
            smt, idx = self._parse_smt(idx)
            smts.append(smt)
        node = ProgramNode(self.tok_region(0, idx), None, smts)
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
        elif self.matches(idx, KwdM('let')) or self.matches(idx, KwdM('global')):
            smt, idx = self._parse_decl(idx)  # Could be faster ^^
        elif self.matches(idx, SemicolonToken):
            smt = NopNode(self.tok_region(idx, idx + 1))
            idx += 1
        else:
            # can only be an expr/(LHS of) assignment
            smt, idx = self._parse_expr_or_assign(idx)
        return smt, idx

    def _parse_expr_or_assign(self, idx: int) -> tuple[AnyNode, int]:
        expr_or_lvalue, idx = self._parse_expr(idx)
        if isinstance(self[idx], SemicolonToken):
            idx += 1
            return expr_or_lvalue, idx
        elif op := self.match_ops(idx, ASSIGN_OPS):
            # TODO multiple assignment?
            idx += 1
            lvalue = expr_or_lvalue
            rvalue, idx = self._parse_expr(idx)
            if self.match_ops(idx, ASSIGN_OPS):
                raise self.err("Multiple assignment is not supported (yet)", self[idx])
            if not isinstance(self[idx], SemicolonToken):
                raise self.err(f"Expected semicolon at end of expr, "
                               f"got {self[idx].name}", self[idx])
            idx += 1
            return self.node_from_children(op, [lvalue, rvalue]), idx
        raise self.err(f"Expected semicolon at end of expr, "
                       f"got {self[idx].name}", self[idx])

    def _parse_decl(self, idx: int):         # e.g. global [] foo=bar, ...;
        scope, idx = self._parse_decl_scope(idx)  # ~~~~~^ ^^ ~~^~~~~~~~~~
        tp_node, idx = self._parse_decl_sqb_or(idx)     # ~/    |
        decl_items, idx = self._parse_decl_item_list(idx)  # ~~~/
        if not self.matches(idx, SemicolonToken):
            raise self.err(f"Expected ';' or ',' after decl_item,"
                           f" got {self[idx].name}", self[idx])
        idx += 1
        return self.node_from_children(DeclNode, [scope, tp_node, decl_items]), idx

    def _parse_decl_scope(self, idx: int) -> tuple[AnyNode, int]:
        if self.matches(idx, KwdM('let')):
            return DeclScope_Let.of(self[idx]), idx + 1
        if self.matches(idx, KwdM('global')):
            return DeclScope_Global.of(self[idx]), idx + 1
        assert 0, "Unknown decl scope"

    def _parse_decl_sqb_or(self, idx: int) -> tuple[AnyNode, int]:
        if not isinstance(self[idx], LSqBracket):
            loc = self[idx - 1].region.end  # Have to give region, so say char after 'let'
            return DeclType_Variable(StrRegion(loc, loc)), idx
        sqb_start = idx
        idx += 1
        if not isinstance(self[idx], RSqBracket):
            raise self.err("Expected ']' after '[' in list decl (eg. 'let[] a;')", self[idx])
        idx += 1
        return DeclType_List(self.tok_region(sqb_start, idx)), idx

    def _parse_decl_item_list(self, idx: int) -> tuple[AnyNode, int]:
        items = []
        while True:
            item, idx = self._parse_decl_item(idx)
            items.append(item)
            if not self.matches(idx, CommaToken):
                break
            idx += 1
        return self.node_from_children(DeclItemsList, items), idx

    def _parse_decl_item(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        if not self.matches(idx, IdentNameToken):
            raise self.err(f"Expected identifier in decl_item, "
                           f"got {self[idx].name}", self[idx])
        children = [node_from_token(self[idx])]
        idx += 1
        # 1. global x, y = ...;
        #            ^
        # 2. global a=..., b;
        #            ^
        if self.matches(idx, OpM('=')):
            # case 2
            idx += 1
            value, idx = self._parse_expr(idx)
            if not isinstance(self.get(idx), (SemicolonToken, CommaToken)):
                raise self.err(f"Expected ';' or ',' after decl_item,"
                               f" got {self[idx].name}", self[idx])
            children.append(value)
        return DeclItemNode(self.tok_region(start, idx), None, children), idx

    def _parse_define(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, KwdM('def'))
        idx += 1
        assert self.matches(idx, IdentNameToken)
        name = node_from_token(self.tokens[idx])
        idx += 1
        args_decl, idx = self._parse_args_decl(idx)
        # def f(t1 arg1, t2 arg2) { <a block> }
        #                         ^
        block, idx = self._parse_block(idx)
        return DefineNode(self.tok_region(start, idx), None, [name, args_decl, block]), idx

    def _parse_args_decl(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        if not self.matches(idx, LParToken):
            raise self.err(f"Expected '(' after 'def name', "
                           f"got {self[idx].name}", self[idx])
        idx += 1
        if self.matches(idx, RParToken):
            # simple case, no args
            idx += 1
            return ArgsDeclNode(self.tok_region(start, idx)), idx
        arg1, idx = self._parse_arg_decl(idx)
        arg_declares = [arg1]
        while not self.matches(idx, RParToken):
            if not self.matches(idx, CommaToken):
                raise self.err(f"Expected ',' or ')' after arg_decl, "
                               f"got {self[idx].name}", self[idx])
            idx += 1
            if self.matches(idx, RParToken):
                # def f(t1 arg1, t2 arg2,)
                #                       ~^
                break
            # def f(t1 arg1, t2 arg2)
            #              ~~^^
            arg, idx = self._parse_arg_decl(idx)
            arg_declares.append(arg)
        # def f(t1 arg1, t2 arg2)
        #                       ^
        assert self.matches(idx, RParToken)
        idx += 1
        return ArgsDeclNode(self.tok_region(start, idx), None, arg_declares), idx

    def _parse_arg_decl(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        if not self.matches(idx, IdentNameToken):
            raise self.err(f"Error: expected type name, got {self[idx].name}."
                           f"Did you forget a ')'?", self[idx])
        tp_name = node_from_token(self[idx])
        idx += 1
        if not self.matches(idx, IdentNameToken):
            raise self.err(f"Error: expected arg name, got {self[idx].name}."
                           f"Did you forget the type name?", self[idx])
        arg_name = node_from_token(self[idx])
        idx += 1
        arg_decl = ArgDeclNode(self.tok_region(start, idx), None, [tp_name, arg_name])
        return arg_decl, idx

    def tok_region(self, start: int, end: int) -> StrRegion:
        start = self.get(start).region.start
        # end is exclusive
        end = self.get(end - 1).region.end
        return StrRegion(start, end)

    def _parse_block(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        if not self.matches(idx, LBrace):
            raise self.err(f"Expected '{{' to start block, "
                           f"got {self[idx].name}", self[idx])
        idx += 1
        smts: list[AnyNode] = []
        while not self.matches(idx, RBrace):
            smt, idx = self._parse_smt(idx)
            smts.append(smt)
        if not self.matches(idx, RBrace):
            raise self.err(f"Expected '}}' to close block, "
                           f"got {self[idx].name}", self[idx])
        idx += 1
        return BlockNode(self.tok_region(start, idx), None, smts), idx

    def _parse_block_with_header(self, start: int, cls: type[AnyNamedNode],
                                 name: str = None) -> tuple[AnyNode, int]:
        name = name or cls.name
        idx = start
        assert self.matches(idx, KwdM(name))
        idx += 1
        expr, idx = self._parse_expr(idx)
        if not self.matches(idx, LBrace):
            raise self.err(f"Expected '{{' after expr in {name}, "
                           f"got {self[idx].name}", self[idx])
        block, idx = self._parse_block(idx)
        return cls(self.tok_region(start, idx), None, [expr, block]), idx

    def _parse_while(self, start: int) -> tuple[AnyNode, int]:
        return self._parse_block_with_header(start, WhileBlock)

    def _parse_repeat(self, start: int) -> tuple[AnyNode, int]:
        return self._parse_block_with_header(start, RepeatBlock)

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
                raise self.err(f"Expected '{{' or 'if' after 'else', "
                               f"got {self[idx + 1].name}", self[idx + 1])
        if else_part is None:
            # Need to give it a location, so just do the '}' (prev token)
            else_part = NullElseBlock(self.tok_region(idx - 1, idx))
        return ConditionalBlock(self.tok_region(start, idx), None,
                                [if_part, *elseif_parts, else_part]), idx

    def _parse_if_cond(self, start: int) -> tuple[AnyNode, int]:
        return self._parse_block_with_header(start, IfBlock, 'if')

    def _parse_elseif(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, (KwdM('else'), KwdM('if')))
        idx += 2
        cond, idx = self._parse_expr(idx)
        if not self.matches(idx, LBrace):
            raise self.err(f"Expected '{{' after expr in else if, "
                           f"got {self[idx].name}", self[idx])
        block, idx = self._parse_block(idx)
        return ElseIfBlock(self.tok_region(start, idx), None, [cond, block]), idx

    def _parse_else(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, (KwdM('else'), LBrace))
        idx += 1  # don't advance past '{'; it's needed for _parse_block
        block, idx = self._parse_block(idx)
        return ElseBlock(self.tok_region(start, idx), None, [block]), idx

    def _parse_call_args(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, LParToken)
        idx += 1
        if self.matches(idx, RParToken):
            # simple case, no args
            idx += 1
            return CallArgs(self.tok_region(start, idx)), idx
        arg1, idx = self._parse_expr(idx)
        args = [arg1]
        while not self.matches(idx, RParToken):
            if not self.matches(idx, CommaToken):
                raise self.err(f"Expected ',' or ')' after arg, got "
                               f"{self[idx].name}", self[idx])
            idx += 1
            if self.matches(idx, RParToken):
                # f(arg1, arg2,)
                #             ~^
                break
            # f(arg1, arg2)
            #       ~~^
            arg, idx = self._parse_expr(idx)
            args.append(arg)
        # f(t1, t2)
        #         ^
        assert self.matches(idx, RParToken)
        idx += 1
        return CallArgs(self.tok_region(start, idx), None, args), idx

    def _parse_expr(self, start: int) -> tuple[AnyNode, int]:
        expr, idx = self._parse_or_bool(start)
        return expr, idx

    def _token_str(self, idx: int):
        return self[idx].get_str(self.src)

    def _expect_cls_consume(
            self, idx: int, cls_or_list: type | tuple[type, ...],
            msg: Exception | str = None, reason: Exception = None) -> int:  # [[nodiscard]]
        if isinstance(self[idx], cls_or_list):
            return idx + 1
        if msg is None:
            msg = self.err(f"Unexpected {self[idx].name}", self[idx])
        elif isinstance(msg, str):
            msg = self.err(msg, self[idx])
        raise msg from reason

    # TODO: maybe add a brk_reason to _parse_expr like 'unrecognized token %'
    def _parse_atom(self, idx: int) -> tuple[AnyNode, int]:
        """Parses literal/ident ('atom' as in can't be broken down further, like 'atomic')"""
        tok = self[idx]
        if isinstance(tok, (StringToken, NumberToken, IdentNameToken)):
            return node_from_token(tok), idx + 1
        raise self.err(f"Unexpected {tok.name} token {tok.get_str(self.src)!r}", tok)

    def _parse_autocat_or_string(self, idx: int) -> tuple[AnyNode, int]:
        start = idx
        strings = []
        while isinstance(self[idx], StringToken):
            strings.append(node_from_token(self[idx]))
            idx += 1
        assert strings, "_parse_autocat_or_string requires current token to be string"
        if len(strings) == 1:
            return strings[0], idx
        return AutocatNode(self.tok_region(start, idx), None, strings), idx

    def _parse_atom_or_autocat(self, idx: int) -> tuple[AnyNode, int]:
        tok = self[idx]
        if isinstance(tok, (NumberToken, IdentNameToken)):
            return node_from_token(tok), idx + 1
        if isinstance(tok, StringToken):
            return self._parse_autocat_or_string(idx)
        raise self.err(f"Unexpected {tok.name} token {tok.get_str(self.src)!r}", tok)

    def _parse_parens_or(self, idx: int) -> tuple[AnyNode, int]:
        start = idx
        if isinstance(self[idx], LParToken):
            inner, idx = self._parse_expr(idx + 1)
            idx = self._expect_cls_consume(
                idx, RParToken, f"Expected ')' at end of expr, got {self[idx].name}")
            return ParenNode(self.tok_region(start, idx), None, [inner]), idx
        elif isinstance(self[idx], LSqBracket):
            return self._parse_list_literal(idx)
        return self._parse_atom_or_autocat(idx)

    def _parse_list_literal(self, idx: int) -> tuple[AnyNode, int]:
        start = idx
        assert self.matches(idx, LSqBracket)
        idx += 1
        if self.matches(idx, RSqBracket):
            idx += 1  # simple case, no args
            return ListNode(self.tok_region(start, idx)), idx
        arg1, idx = self._parse_expr(idx)
        args = [arg1]
        while not self.matches(idx, RSqBracket):
            if not self.matches(idx, CommaToken):  # if not end, must be comma
                raise self.err(f"Expected ',' or ']' after item in list"
                               f" literal, got {self[idx].name}", self[idx])
            idx += 1
            if self.matches(idx, RSqBracket):  # trailing comma, no value
                break
            arg, idx = self._parse_expr(idx)
            args.append(arg)
        assert self.matches(idx, RSqBracket)
        idx += 1
        return ListNode(self.tok_region(start, idx), None, args), idx

    def _parse_basic_item(self, idx: int):
        left, new_idx = self._parse_parens_or(idx)
        while idx != (idx := new_idx):  # If progress made, set old to current and loop again
            left, new_idx = self._parse_basic_item_chain_once(idx, left)
        return left, new_idx

    def _parse_basic_item_chain_once(self, idx: int, left: AnyNode) -> tuple[AnyNode, int]:
        if isinstance(self[idx], DotToken):
            idx += 1
            if not isinstance(self[idx], AttrNameToken):
                raise self.err(f"Expected attribute name after '.', "
                               f"got {self[idx].name}", self[idx])
            right = node_from_token(self[idx])
            idx += 1
            return self.node_from_children(GetattrNode, [left, right]), idx
        elif isinstance(self[idx], LSqBracket):
            idx += 1
            inner, idx = self._parse_expr(idx)
            if not isinstance(self[idx], RSqBracket):
                raise self.err(f"Expected rsqb, got {self[idx].name}", self[idx])
            node = self.node_from_children(GetitemNode, [left, inner],
                                           region=[left, inner, self[idx]])
            idx += 1
            return node, idx
        elif isinstance(self[idx], LParToken):
            args, idx = self._parse_call_args(idx)
            return self.node_from_children(CallNode, [left, args]), idx
        return left, idx

    def match_ops(self, idx: int, *ops: str | Sequence[str]) -> str | None:
        tok = self[idx]
        if not isinstance(tok, OpToken):
            return None
        for op in ops:
            if isinstance(op, str):
                if tok.op_str == op:
                    return op
            else:
                if value := self.match_ops(idx, *op):
                    return value
        return None

    def _parse_unaries_into_tok_list(self, idx: int) -> tuple[list[OpToken], int]:
        ls = []
        while self.match_ops(idx, UNARY_OPS):
            ls.append(cast(OpToken, self[idx]))
            idx += 1
        return ls, idx

    def _apply_unaries_list(self, unaries_list: list[OpToken], inner: AnyNode):
        curr = inner
        for tok in reversed(unaries_list):
            curr = self.node_from_children(tok.op_str, [curr], region=[tok, curr])
        return curr

    def _parse_pow_or(self, idx: int) -> tuple[AnyNode, int]:
        leftmost, idx = self._parse_basic_item(idx)
        if not self.match_ops(idx, '**'):
            return leftmost, idx
        idx += 1
        # parse_unary_or -> parse_pow_or -> parse_unary_or -> parse_pow_or
        # This is right-recursion so is fine as progress will be made each call to get here
        right, idx = self._parse_unary_or(idx)
        return self.node_from_children(PowNode, [leftmost, right]), idx

    def _parse_unary_or(self, idx: int) -> tuple[AnyNode, int]:
        unaries, idx = self._parse_unaries_into_tok_list(idx)
        inner, idx = self._parse_pow_or(idx)
        return self._apply_unaries_list(unaries, inner), idx

    def _parse_ltr_operator_level(
            self, idx: int, ops: Iterable[str],
            inner_level: Callable[[int], tuple[AnyNode, int]]) -> tuple[AnyNode, int]:
        curr, idx = inner_level(idx)
        while self.match_ops(idx, *ops):
            op = cast(OpToken, self[idx]).op_str
            idx += 1
            right, idx = inner_level(idx)
            curr = self.node_from_children(op, [curr, right])
        return curr, idx

    def _parse_mul_div(self, idx: int) -> tuple[AnyNode, int]:
        return self._parse_ltr_operator_level(idx, ('*', '/', '%'), self._parse_unary_or)

    def _parse_add_sub(self, idx: int) -> tuple[AnyNode, int]:
        return self._parse_ltr_operator_level(idx, ('+', '-'), self._parse_mul_div)

    def _parse_cat(self, idx: int) -> tuple[AnyNode, int]:
        return self._parse_ltr_operator_level(idx, ('..',), self._parse_add_sub)

    def _parse_comp(self, idx: int) -> tuple[AnyNode, int]:
        first, idx = self._parse_cat(idx)
        parts = [first]
        while self.match_ops(idx, COMPARISONS):
            op_tok = cast(OpToken, self[idx])
            idx += 1
            curr, idx = self._parse_cat(idx)
            parts += [op_tok, curr]
        if len(parts) == 1:
            return parts[0], idx
        assert len(parts) % 2 == 1
        if len(parts) > 3:
            # TODO: chained comparisons
            raise self.err("Chaining comparisons is not yet supported", parts[3])
        left, op_tok, right = parts
        return self.node_from_children(op_tok.op_str, [left, right]), idx

    def _parse_and_bool(self, idx: int):
        return self._parse_ltr_operator_level(idx, ('&&',), self._parse_comp)

    def _parse_or_bool(self, idx: int) -> tuple[AnyNode, int]:
        return self._parse_ltr_operator_level(idx, ('||',), self._parse_and_bool)

    def err(self, msg: str, loc: RegionUnionArgT):
        return LocatedCstError(msg, region_union(loc), self.src)

    @classmethod
    def node_from_children(cls, name_or_type: str | type[AnyNamedNode],
                           children: list[AnyNode],
                           region: RegionUnionArgT = None,
                           parent: Node = None, arity: int = None):
        region = region_union(region if region is not None else children)
        if isinstance(name_or_type, str):
            klass = node_cls_from_name(name_or_type, children, arity)
        else:
            klass = name_or_type
        return klass(region, parent, children)


# operator precedence (most to least binding):
#  1. () [] {}  (parens)
#  2. string literal auto-concatenation ("ab" "xy" => "abxy")
#  3. . [] fn_call() (getattr, getitem, function call)
#  4.
#  4.1 **-
#  4.2 ** (bare)
#  5. + - ! (unary)
#  6. * / %
#  7. + -
#  8. ..
#  9. == != < > <= >=
# 10. &&
# 11. ||

# Note: level 1 and 2 may be swapped
