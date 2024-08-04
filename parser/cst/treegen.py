from __future__ import annotations

from typing import (TypeVar, cast, TypeAlias, Sequence, overload)

from parser.cst.token_matcher import OpM, KwdM, Matcher, PatternT
from parser.error import BaseParseError, BaseLocatedError
from parser.lexer import Tokenizer
from parser.operators import UNARY_OPS, COMPARISONS, ASSIGN_OPS
from parser.str_region import StrRegion
from parser.tokens import *
from parser.tree_node import Node, Leaf, AnyNode

DT = TypeVar('DT')
ET = TypeVar('ET')

MISSING = object()

KEYWORDS = ['def', 'if', 'else', 'while', 'repeat', 'global', 'let']


class CstParseError(BaseParseError):
    pass


class LocatedCstError(BaseLocatedError, CstParseError):
    pass


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
        elif self.matches(idx, SemicolonToken):
            smt = Leaf('nop_smt', self.tok_region(idx, idx + 1))
            idx += 1
        else:
            # can only be an expr
            # todo for now we are assuming that a smt can contain any expr
            #  but this could/will change in the future
            #  although it may be better not to deal with it here
            #  and instead do a post-processing step
            # TODO: maybe include ';' in the inner - but why?
            #  Need an smt type which also includes the ';'
            smt, idx = self._parse_expr_or_assign(idx)
        return smt, idx

    # TODO: recheck all self[idx] stuff as we may get trouble with eofs
    def _parse_expr_or_assign(self, idx: int) -> tuple[AnyNode, int]:
        expr_or_lvalue, idx, legacy_brk_reason = self._parse_expr(idx)
        assert not legacy_brk_reason
        if isinstance(self[idx], SemicolonToken):
            idx += 1
            return expr_or_lvalue, idx  # TODO: maybe have a smt node?
        elif op := self.match_ops(idx, ASSIGN_OPS):
            # TODO multiple assignment?
            idx += 1
            lvalue = expr_or_lvalue
            rvalue, idx, legacy_brk_reason = self._parse_expr(idx)
            assert not legacy_brk_reason
            if self.match_ops(idx, ASSIGN_OPS):
                raise self.err("Multiple assignment is not supported (yet)", self[idx])
            if not isinstance(self[idx], SemicolonToken):
                raise self.err("Expected semicolon at end of expr", self[idx])
            idx += 1
            return self.node_from_children(op, [lvalue, rvalue]), idx
        raise self.err("Expected semicolon at end of expr", self[idx])

    def _parse_let(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, KwdM('let'))
        idx += 1
        items, idx = self._parse_decl_item_list(idx)
        if not self.matches(idx, SemicolonToken):
            raise self.err(f"Expected ';' or ',' after decl_item,"
                           f" got {self[idx].name}", self[idx])
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
            raise self.err(f"Expected ';' or ',' after decl_item,"
                           f" got {self[idx].name}", self[idx])
        idx += 1
        return Node('global_decl', self.tok_region(start, idx),
                    None, items), idx

    def _parse_decl_item(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        if not self.matches(idx, IdentNameToken):
            raise self.err(f"Expected identifier in decl_item, "
                           f"got {self[idx].name}", self[idx])
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
                raise self.err(f"Expected ';' or ',' after decl_item,"
                               f" got {self[idx].name}", self[idx]) from brk_reason
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
            raise self.err("Expected '(' after 'def name'", self[idx])
        idx += 1
        if self.matches(idx, RParToken):
            # simple case, no args
            idx += 1
            return Node('args_decl', self.tok_region(start, idx)), idx
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
        # def f(t1, t2)
        #             ^
        assert self.matches(idx, RParToken)
        idx += 1
        args_decl = Node('args_decl', self.tok_region(start, idx), None, arg_declares)
        return args_decl, idx

    def _parse_arg_decl(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        if not self.matches(idx, IdentNameToken):
            raise self.err(f"Error: expected type name, got {self[idx].name}."
                           f"Did you forget a ')'?", self[idx])
        tp_name = Leaf.of(self[idx])
        idx += 1
        if not self.matches(idx, IdentNameToken):
            raise self.err(f"Error: expected arg name, got {self[idx].name}."
                           f"Did you forget the type name?", self[idx])
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
        return Node('block', self.tok_region(start, idx), None, smts), idx

    def _parse_while(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, KwdM('while'))
        idx += 1
        cond, idx, brk_reason = self._parse_expr(idx)
        if not self.matches(idx, LBrace):
            raise self.err("Expected '{' after expr in while", self[idx]) from brk_reason
        block, idx = self._parse_block(idx)
        return Node('while', self.tok_region(start, idx),
                    None, [cond, block]), idx

    def _parse_repeat(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, KwdM('repeat'))
        idx += 1
        amount, idx, brk_reason = self._parse_expr(idx)
        if not self.matches(idx, LBrace):
            raise self.err("Expected '{' after expr in repeat", self[idx]) from brk_reason
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
                raise self.err(f"Expected '{{' or 'if' after 'else', "
                               f"got {self[idx + 1].name}", self[idx + 1])
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
            raise self.err("Expected '{' after expr in if", self[idx]) from brk_reason
        block, idx = self._parse_block(idx)
        return Node('if_cond', self.tok_region(start, idx),
                    None, [cond, block]), idx

    def _parse_elseif(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, (KwdM('else'), KwdM('if')))
        idx += 2
        cond, idx, brk_reason = self._parse_expr(idx)
        if not self.matches(idx, LBrace):
            raise self.err("Expected '{' after expr in else if",
                           self[idx]) from brk_reason
        block, idx = self._parse_block(idx)
        return Node('elseif_cond', self.tok_region(start, idx),
                    None, [cond, block]), idx

    def _parse_else(self, start: int) -> tuple[AnyNode, int]:
        idx = start
        assert self.matches(idx, (KwdM('else'), LBrace))
        idx += 1  # don't advance past '{'; it's needed for _parse_block
        block, idx = self._parse_block(idx)
        return Node('else_cond', self.tok_region(start, idx), None, [block]), idx

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
                raise self.err(f"Expected ',' or ')' after arg, got "
                               f"{self[idx].name}", self[idx]) from brk_reason
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

    def _parse_expr(self, start: int, partial: AnyNode = None,
                    partial_end: int = None) -> tuple[AnyNode, int, BaseParseError | None]:
        if partial or partial_end:
            raise NotImplementedError
        expr, idx = self._parse_or_bool(start)
        return expr, idx, None

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

    r"""
    Basic grammar:
    '(' = LPAR
    ')' = RPAR
    '[' = LSQB
    ']' = RSQB
    
    '.' = DOT
    ',' = COMMA
    
    '+' < OP
    '-' < OP
    '*' < OP
    '/' < OP
    '%' < OP
    '**' < OP
    
    '!' < OP
    '&&' < OP
    '||' < OP
    
    '..' < OP
    
    '<' < OP
    '>' < OP
    '<=' < OP
    '>=' < OP
    '==' < OP
    '!=' < OP
    
    Unary = '+' | '-' | '!'
    Mul_Div = '*' | '/'
    Add_Sub = '+' | '-'
    Comp = '<' | '>' | '<=' | '>=' | '==' | '!='
    
    # -- Expressions --
    atom := STRING | NUMBER | IDENT
    
    autocat := (STRING)+
    atom_or_autocat := atom | autocat                                  # level 2
       # \- implemented as NUMBER | IDENT | (STRING | STRING+)
    
    parens_or := '(' expr ')' | atom_or_autocat                        # level 1
    
    basic_item := parens_or (basic_item_chain)*                        # level 3
    basic_item_chain := fn_call_chain | getattr_chain | getitem_chain
    fn_call_chain := '(' fn_args ')'
    getattr_chain := '.' ATTR_NAME
    getitem_chain := '[' expr ']'
       # \ - could be:
       #   basic_item := fn_call | getattr | getitem | basic_item
       #   fn_call := basic_item '(' fn_args ')'
       #   getattr := basic_item '.' ATTR_NAME
       #   getitem := basic_item '[' expr ']'
       # But that could be trapped in infinite loop due to left-recursion:
       #  parse basic_item -> try fn_call -> parse basic_item -> ...
    fn_args := expr (',' expr)* (',')?
    
    unary_pow_rhs := (Unary)* basic_item             # level 4.1
    @grouping:rtl
    pow_or := basic_item ('**' unary_pow_rhs)*       # level 4.2
    unary_or := (Unary)* pow_or                      # level 5
    mul_div_or := unary_or (Mul_Div unary_or)*       # level 6
    add_sub_or := mul_div_or (Add_Sub mul_div_or)*   # level 7
    cat_or := add_sub_or ('..' add_sub_or)*          # level 8
    @grouping:special
    comp_or := cat_or (Comp cat_or)*                 # level 9
    and_bool_or := comp_or ('&&' comp_or)*           # level 10
    or_bool_or := and_bool_or ('||' and_bool_or)*    # level 11
    
    expr := or_bool_or
    
    ...  # (smt, etc.)
    
    """
    def _parse_atom(self, idx: int) -> tuple[AnyNode, int]:
        """Parses literal/ident ('atom' as in can't be broken down further, like 'atomic')"""
        tok = self[idx]
        if isinstance(tok, (StringToken, NumberToken, IdentNameToken)):
            return Leaf.of(tok), idx + 1
        raise self.err(f"Unexpected {tok.name} token {tok.get_str(self.src)!r}", tok)

    def _parse_autocat_or_string(self, idx: int) -> tuple[AnyNode, int]:
        start = idx
        strings = []
        while isinstance(self[idx], StringToken):
            strings.append(Leaf.of(self[idx]))
            idx += 1
        assert strings, "_parse_autocat_or_string requires current token to be string"
        if len(strings) == 1:
            return strings[0], idx
        return Node('autocat', self.tok_region(start, idx), None, strings), idx

    def _parse_atom_or_autocat(self, idx: int) -> tuple[AnyNode, int]:
        tok = self[idx]
        if isinstance(tok, (NumberToken, IdentNameToken)):
            return Leaf.of(tok), idx + 1
        if isinstance(tok, StringToken):
            return self._parse_autocat_or_string(idx)
        raise self.err(f"Unexpected {tok.name} token {tok.get_str(self.src)!r}", tok)

    def _parse_parens_or(self, idx: int) -> tuple[AnyNode, int]:
        start = idx
        if isinstance(self[idx], LParToken):
            inner, idx, brk_reason_outdated = self._parse_expr(idx + 1)
            idx = self._expect_cls_consume(
                idx, RParToken, "Expected ')' at end of expr", brk_reason_outdated)
            return Node('paren', self.tok_region(start, idx), None, [inner]), idx
        return self._parse_atom_or_autocat(idx)

    def _parse_basic_item(self, idx: int):
        old_idx = -999  # to start the loop
        left, idx = self._parse_parens_or(idx)
        while idx != old_idx:
            old_idx = idx
            left, idx = self._parse_basic_item_chain_once(idx, left)
        return left, idx

    def _parse_basic_item_chain_once(self, idx: int, left: AnyNode) -> tuple[AnyNode, int]:
        if isinstance(self[idx], DotToken):
            idx += 1
            if not isinstance(self[idx], AttrNameToken):
                raise self.err("Expected attribute name after '.'", self[idx])
            right = Leaf.of(self[idx])
            idx += 1
            return self.node_from_children('getattr', [left, right]), idx
        elif isinstance(self[idx], LSqBracket):
            idx += 1
            inner, idx, legacy_brk_reason = self._parse_expr(idx)
            if not isinstance(self[idx], RSqBracket):
                raise self.err(f"Expected rsqb, got {self[idx].name}",
                               self[idx]) from legacy_brk_reason
            node = self.node_from_children('getitem', [left, inner],
                                           region=[left, inner, self[idx]])
            idx += 1
            return node, idx
        elif isinstance(self[idx], LParToken):
            args, idx = self._parse_call_args(idx)
            return self.node_from_children('call', [left, args]), idx
        return left, idx

    def match_ops(self, idx: int, *ops: str | Sequence[str]):
        tok = self[idx]
        if not isinstance(tok, OpToken):
            return False
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
        return self._parse_pow_rhs_item(idx)

    def _parse_pow_rhs_item(self, idx: int) -> tuple[AnyNode, int]:
        unaries, idx = self._parse_unaries_into_tok_list(idx)
        # This is right-recursion so is fine as progress will be made each call to get here
        inner, idx = self._parse_pow_or(idx)
        return self._apply_unaries_list(unaries, inner), idx

    def _parse_unary_or(self, idx: int) -> tuple[AnyNode, int]:
        unaries, idx = self._parse_unaries_into_tok_list(idx)
        inner, idx = self._parse_pow_or(idx)
        return self._apply_unaries_list(unaries, inner), idx

    # TODO these are very similar - unify them?
    def _parse_mul_div(self, idx: int) -> tuple[AnyNode, int]:
        curr, idx = self._parse_unary_or(idx)
        while self.match_ops(idx, '*', '/'):
            op = cast(OpToken, self[idx]).op_str
            idx += 1
            right, idx = self._parse_unary_or(idx)
            curr = self.node_from_children(op, [curr, right])
        return curr, idx

    def _parse_add_sub(self, idx: int) -> tuple[AnyNode, int]:
        curr, idx = self._parse_mul_div(idx)
        while self.match_ops(idx, '+', '-'):
            op = cast(OpToken, self[idx]).op_str
            idx += 1
            right, idx = self._parse_mul_div(idx)
            curr = self.node_from_children(op, [curr, right])
        return curr, idx

    def _parse_cat(self, idx: int) -> tuple[AnyNode, int]:
        curr, idx = self._parse_add_sub(idx)
        while self.match_ops(idx, '..'):
            idx += 1
            right, idx = self._parse_add_sub(idx)
            curr = self.node_from_children('..', [curr, right])
        return curr, idx

    def _parse_comp(self, idx: int) -> tuple[AnyNode, int]:
        first, idx = self._parse_cat(idx)
        parts = [first]
        while self.match_ops(idx, COMPARISONS):
            op = cast(OpToken, self[idx]).op_str
            idx += 1
            curr, idx = self._parse_cat(idx)
            parts += [op, curr]
        if len(parts) == 1:
            return parts[0], idx
        assert len(parts) % 2 == 1
        if len(parts) > 3:
            # TODO: chained comparisons
            raise self.err("Chaining comparisons is not yet supported", parts[3])
        left, op, right = parts
        return self.node_from_children(op, [left, right]), idx

    def _parse_and_bool(self, idx: int):
        curr, idx = self._parse_comp(idx)
        while self.match_ops(idx, '&&'):
            idx += 1
            right, idx = self._parse_comp(idx)
            curr = self.node_from_children('&&', [curr, right])
        return curr, idx

    def _parse_or_bool(self, idx: int) -> tuple[AnyNode, int]:
        curr, idx = self._parse_and_bool(idx)
        while self.match_ops(idx, '||'):
            idx += 1
            right, idx = self._parse_and_bool(idx)
            curr = self.node_from_children('||', [curr, right])
        return curr, idx

    def err(self, msg: str, loc: RegionUnionArgT):
        return LocatedCstError(msg, self.region_union(loc), self.src)

    @classmethod
    def region_union(cls, *args: RegionUnionArgT):
        regs = []
        for loc in args:
            if isinstance(loc, (Token, AnyNode)):
                loc = loc.region  # Token and AnyNode, both have `.region`
            if isinstance(loc, StrRegion):
                regs.append(loc)
            else:
                regs.append(cls.region_union(*loc))
        return StrRegion.union(*regs)

    @classmethod
    def node_from_children(cls, name: str, children: list[AnyNode],
                           region: RegionUnionArgT = None, parent: Node = None):
        region = cls.region_union(region if region is not None else children)
        return Node(name, region, parent, children)


CstGen = TreeGen

RegionUnionFlatT: TypeAlias = 'Token | AnyNode | StrRegion'
RegionUnionArgT: TypeAlias = 'RegionUnionFlatT | Sequence[RegionUnionFlatT]'

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

# Note: level 1 and 2 may be swapped


def _seq_get(self: Sequence[ET], item: int, default: DT) -> ET | DT:
    try:
        return self[item]
    except IndexError:
        return default
