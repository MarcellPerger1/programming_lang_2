import unittest

from parser.lexer.tokenizer import Tokenizer
from parser.operators import BINARY_OPS
from parser.cst.treegen import TreeGen, CstParseError
from parser.cst.tree_print import tprint

from test.snapshottest import SnapshotTestCase


class TreeGenTest(SnapshotTestCase):
    maxDiff = None

    def test_item_chain(self):
        tk = Tokenizer('a[7].b.0.fn["c" .. 2] = fn(9).k[7 + r](3,);')
        t = TreeGen(tk)
        t.parse()
        tprint(t.result)
        self.assertMatchesSnapshot(t.result)

    def test_fn_call_in_lvalue(self):
        tk = Tokenizer('a(7).b.0.fn()["c" .. 2] = fn(9).k[7 + r](3,);')
        t = TreeGen(tk)
        t.parse()
        tprint(t.result)
        self.assertMatchesSnapshot(t.result)

    def test_aug_assign(self):
        tk = Tokenizer('a[1] += a.2;')
        t = TreeGen(tk)
        t.parse()
        tprint(t.result)
        self.assertMatchesSnapshot(t.result)

    # noinspection PyMethodMayBeStatic
    def assertValidParse(self, src: str):
        t = TreeGen(Tokenizer(src))
        t.parse()

    @unittest.expectedFailure  # # TODO!!!!!!!!! I have forgotten about '%' !!!
    def test__mode_supported(self):
        # TODO: check output
        self.assertValidParse(f'c=a%b;')


class TreeGenEofTest(SnapshotTestCase):
    def test__at_expr_end(self):
        self.assertFailsGracefully('a.b()')

    def assertFailsGracefully(self, src: str):
        t = TreeGen(Tokenizer(src))
        with self.assertRaises(CstParseError):
            t.parse()

    # noinspection PyMethodMayBeStatic
    def assertValidParse(self, src: str):
        t = TreeGen(Tokenizer(src))
        t.parse()

    def test__at_assign_end(self):
        self.assertFailsGracefully('a.b = 8')

    def test__at_let_end(self):
        self.assertFailsGracefully('let a = 3')

    def test__at_global_end(self):
        self.assertFailsGracefully('global a = 3')

    def test__at_decl_item_begin(self):
        self.assertFailsGracefully('let')

    @unittest.expectedFailure  # TODO fix this
    def test__at_decl_item_could_be_eq(self):
        self.assertFailsGracefully('let a')

    def test__at_decl_item_after_eq(self):
        self.assertFailsGracefully('let a=')

    def test__at_def_start(self):
        self.assertFailsGracefully('def')

    def test__at_def_name(self):
        self.assertFailsGracefully('def a')

    def test__at_def_lpar(self):
        self.assertFailsGracefully('def a(')

    def test__at_def_param_type(self):
        self.assertFailsGracefully('def a(t1')

    def test__at_def_param_value(self):
        self.assertFailsGracefully('def a(t1 name')

    def test__at_def_param_comma(self):
        self.assertFailsGracefully('def a(t1 name,')

    def test__at_def_param_second_type(self):
        self.assertFailsGracefully('def a(t1 name,t2')

    def test__at_def_param_second_name(self):
        self.assertFailsGracefully('def a(t1 name,t2 name2')

    def test__at_def_param_second_comma(self):
        self.assertFailsGracefully('def a(t1 name,t2 name2,')

    def test__at_def_rpar_noargs(self):
        self.assertFailsGracefully('def a()')

    def test__at_def_rpar_1_arg_comma(self):
        self.assertFailsGracefully('def a(t1 name,)')

    def test__at_def_rpar_1_arg_no_comma(self):
        self.assertFailsGracefully('def a(t1 name)')

    def test__at_def_rpar_2_arg(self):
        self.assertFailsGracefully('def a(t1 name,t2 name2)')

    def test__at_def_lbrace(self):
        self.assertFailsGracefully('def a(t1 name,t2 name2){')

    def test__at_def_rbrace(self):
        self.assertValidParse('def a(t1 name,t2 name2){}')

    def test__at_block_no_semi(self):
        self.assertFailsGracefully('def a(t1 name,t2 name2){let a=1')

    def test__at_block_semi(self):
        self.assertFailsGracefully('def a(t1 name,t2 name2){let a=1;')

    def test__common_flow_kwd(self):
        for name in ('while', 'repeat', 'while'):
            with self.subTest(name=name):
                self.assertFailsGracefully(f'{name}')

    def test__common_flow_cond(self):
        for name in ('while', 'repeat', 'while'):
            with self.subTest(name=name):
                self.assertFailsGracefully(f'{name} a+1')

    def test__common_flow_lbrace(self):
        for name in ('while', 'repeat', 'while'):
            with self.subTest(name=name):
                self.assertFailsGracefully(f'{name} a + 1 {{')

    def test__if_full(self):
        self.assertValidParse('if a+1{}')

    def test__else_or_elseif__else(self):
        self.assertFailsGracefully('if 1{}else')

    def test__elseif_else_if(self):
        self.assertFailsGracefully('if 1{}else if')

    def test__elseif_expr(self):
        self.assertFailsGracefully('if 1{}else if 2')

    def test__elseif_lbrace(self):
        self.assertFailsGracefully('if 1{}else if 2{')

    def test__else_lbrace(self):
        self.assertFailsGracefully('if 1{}else {')

    def test__call_args_lpar(self):
        self.assertFailsGracefully('a(')

    def test__call_args_arg1(self):
        self.assertFailsGracefully('a(a')

    def test__call_args_arg1_comma(self):
        self.assertFailsGracefully('a(a,')

    def test__call_args_comma_rpar(self):
        self.assertFailsGracefully('a(a,)')

    def test__call_args_rpar(self):
        self.assertFailsGracefully('a(a)')

    def test__call_args_noargs(self):
        self.assertFailsGracefully('a()')

    def test__string_single(self):
        self.assertFailsGracefully('"a"')

    def test__string_autocat(self):
        self.assertFailsGracefully('"a""b"')

    def test__number_single(self):
        self.assertFailsGracefully('5')

    def test__lpar(self):
        self.assertFailsGracefully('(')

    def test__lpar_expr(self):
        self.assertFailsGracefully('(a')

    def test__basic_chain_ident_only(self):
        self.assertFailsGracefully('a')

    def test__basic_chain_dot(self):
        self.assertFailsGracefully('a.')

    def test__basic_chain_dot_attr(self):
        self.assertFailsGracefully('a.b')

    def test__basic_chain_lsqb(self):
        self.assertFailsGracefully('a[')

    def test__basic_chain_getitem_expr(self):
        self.assertFailsGracefully('a[b')

    def test__basic_chain_rsqb_full(self):
        self.assertFailsGracefully('a[b]')

    def test__basic_chain_rsqb_empty(self):
        self.assertFailsGracefully('a[]')  # TODO: check that expt '[]' give syntax error

    def test__basic_chain_call_lpar(self):
        self.assertFailsGracefully('a(')

    def test__basic_chain_call_inner(self):
        self.assertFailsGracefully('a(b')

    def test__basic_chain_call_rpar_full(self):
        self.assertFailsGracefully('a(b)')

    def test__basic_chain_call_rpar_empty(self):
        self.assertFailsGracefully('a()')

    def test__unaries_single(self):
        self.assertFailsGracefully('+')

    def test__unaries_many(self):
        self.assertFailsGracefully('-+--!')

    def test__unaries_many_expr(self):
        self.assertFailsGracefully('-+--!a')

    def test__pow_op(self):
        self.assertFailsGracefully('a**')

    def test__pow_rhs_unary(self):
        self.assertFailsGracefully('a**-')

    def test__pow_rhs_unary_expr(self):
        self.assertFailsGracefully('a**-a')

    def test__pow_rhs_expr(self):
        self.assertFailsGracefully('a**b')

    def test__bin_op__op(self):
        for op in BINARY_OPS:
            with self.subTest(op=op):
                self.assertFailsGracefully(f'a{op}')

    def test__bin_op__expr(self):
        for op in BINARY_OPS:
            with self.subTest(op=op):
                self.assertFailsGracefully(f'a{op}b')

    def test__bin_op__success(self):
        for op in set(BINARY_OPS) - {'%'}:  # TODO: '%' failing
            with self.subTest(op=op):
                self.assertValidParse(f'a{op}b;')


if __name__ == '__main__':
    unittest.main()
