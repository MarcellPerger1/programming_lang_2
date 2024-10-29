import unittest

from parser.lexer.tokenizer import Tokenizer
from parser.operators import BINARY_OPS
from parser.cst.treegen import TreeGen, LocatedCstError
from test.common import CommonTestCase


class TreeGenTest(CommonTestCase):
    def test_item_chain(self):
        self.assertTreeMatchesSnapshot('a[7].b.0.fn["c" .. 2] = fn(9).k[7 + r](3,);')

    def test_fn_call_in_lvalue(self):
        self.assertTreeMatchesSnapshot('a(7).b.0.fn()["c" .. 2] = fn(9).k[7 + r](3,);')

    def test_aug_assign(self):
        self.assertTreeMatchesSnapshot('a[1] += a.2;')

    def test__mod_supported(self):
        self.assertTreeMatchesSnapshot('c=a%b;')

    def test_decl_no_value(self):
        self.assertTreeMatchesSnapshot('let b;')


class TestBlocks(CommonTestCase):
    def test_while(self):
        self.assertTreeMatchesSnapshot('while a || !b && c >= 6 {}')
        self.assertTreeMatchesSnapshot('while!(7%8){(7.7).abc(6,7,8);}')

    def test_repeat(self):
        self.assertTreeMatchesSnapshot('repeat a || !b && c >= 6 {}')
        self.assertTreeMatchesSnapshot('repeat!(7%8){(7.7).abc(6,7,8);}')

    def test_else_if_else(self):
        self.assertTreeMatchesSnapshot('if(1){}else if(a||!b&&c!=6){}')
        self.assertTreeMatchesSnapshot('if(1){}else{a();}')
        self.assertTreeMatchesSnapshot('if(1){}else if 9{a();}else{b(a, a());}')


class TestFunctionDecl(CommonTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.setProperCwd()

    def test_no_params(self):
        self.assertTreeMatchesSnapshot('def a() { alert("Called"); }')

    def test_one_param(self):
        self.assertTreeMatchesSnapshot('def a(number val){print(val);}')

    def test_two_param(self):
        self.assertTreeMatchesSnapshot('def a(number a, string b){RESULT=a.."="..b;}')


class TestTreeGenErrors(CommonTestCase):
    def test_empty_sqb_error(self):
        with self.assertRaises(LocatedCstError) as err:
            TreeGen(Tokenizer('v=a[]+b')).parse()
        exc = err.exception
        self.assertBetweenIncl(3, 4, exc.region.start)
        self.assertEqual(4, exc.region.end - 1)


class TreeGenEofTest(CommonTestCase):
    def test__at_expr_end(self):
        self.assertFailsGracefully('a.b()')

    def test__at_assign_end(self):
        self.assertFailsGracefully('a.b = 8')

    def test__at_let_end(self):
        self.assertFailsGracefully('let a = 3')

    def test__at_global_end(self):
        self.assertFailsGracefully('global a = 3')

    def test__at_decl_item_begin(self):
        self.assertFailsGracefully('let')

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
        self.assertFailsGracefully('a[]')

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
        for op in set(BINARY_OPS):
            with self.subTest(op=op):
                self.assertValidParse(f'a{op}b;')


if __name__ == '__main__':
    unittest.main()
