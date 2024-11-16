import unittest

from parser.lexer.tokenizer import Tokenizer
from parser.operators import BINARY_OPS
from parser.cst.treegen import TreeGen, LocatedCstError
from parser.common import StrRegion
from test.common import CommonTestCase


class TestAutocat(CommonTestCase):
    def test_autocat(self):
        self.assertCstMatchesSnapshot('"abc" "="\n  "1" .. a .. "b".d();')


class TreeGenTest(CommonTestCase):
    def test_item_chain(self):
        self.assertCstMatchesSnapshot('a[7].b.0.fn["c" .. 2] = fn(9).k[7 + r](3,);')

    def test_fn_call_in_lvalue(self):
        self.assertCstMatchesSnapshot('a(7).b.0.fn()["c" .. 2] = fn(9).k[7 + r](3,);')

    def test_aug_assign(self):
        self.assertCstMatchesSnapshot('a[1] += a.2;')

    def test__mod_supported(self):
        self.assertCstMatchesSnapshot('c=a%b;')

    def test_decl_no_value(self):
        self.assertCstMatchesSnapshot('let b;')

    def test_decl_multiple(self):
        self.assertCstMatchesSnapshot(
            'let a, b=9, c,d=w.1[2],e="w",f,g;\n'
            'global z,y=-11, x , w="a".lower() ,v=o , u, t;')


class TestBlocks(CommonTestCase):
    def test_while(self):
        self.assertCstMatchesSnapshot('while a || !b && c >= 6 {}')
        self.assertCstMatchesSnapshot('while!(7%8){(7.7).abc(6,7,8);}')

    def test_repeat(self):
        self.assertCstMatchesSnapshot('repeat a || !b && c >= 6 {}')
        self.assertCstMatchesSnapshot('repeat!(7%8){(7.7).abc(6,7,8);}')

    def test_else_if_else(self):
        self.assertCstMatchesSnapshot('if(1){}else if(a||!b&&c!=6){}')
        self.assertCstMatchesSnapshot('if(1){}else{a();}')
        self.assertCstMatchesSnapshot('if(1){}else if 9{a();}else{b(a, a());}')

    def test_else_cond_null(self):
        src = 'if 0==1{exit();  } startup();'
        n = TreeGen(Tokenizer(src)).parse()
        node = n.children[0].children[-1]
        self.assertLessEqual(node.region.start, node.region.end)
        self.assertEqual(StrRegion(17, 18), node.region)


class TestFunctionDecl(CommonTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.setProperCwd()

    def test_no_params(self):
        self.assertCstMatchesSnapshot('def a() { alert("Called"); }')

    def test_one_param(self):
        self.assertCstMatchesSnapshot('def a(number val){print(val);}')

    def test_two_param(self):
        self.assertCstMatchesSnapshot('def a(number a, string b){RESULT=a.."="..b;}')


class TestTreeGenErrors(CommonTestCase):
    def test_empty_sqb_error(self):
        with self.assertRaises(LocatedCstError) as err:
            TreeGen(Tokenizer('v=a[]+b')).parse()
        exc = err.exception
        self.assertBetweenIncl(3, 4, exc.region.start)
        self.assertEqual(4, exc.region.end - 1)


class TreeGenEofTest(CommonTestCase):
    def test__at_expr_end(self):
        self.assertFailsGracefullyCST('a.b()')

    def test__at_assign_end(self):
        self.assertFailsGracefullyCST('a.b = 8')

    def test__at_let_end(self):
        self.assertFailsGracefullyCST('let a = 3')

    def test__at_global_end(self):
        self.assertFailsGracefullyCST('global a = 3')

    def test__at_decl_item_begin(self):
        self.assertFailsGracefullyCST('let')

    def test__at_decl_item_could_be_eq(self):
        self.assertFailsGracefullyCST('let a')

    def test__at_decl_item_after_eq(self):
        self.assertFailsGracefullyCST('let a=')

    def test__at_def_start(self):
        self.assertFailsGracefullyCST('def')

    def test__at_def_name(self):
        self.assertFailsGracefullyCST('def a')

    def test__at_def_lpar(self):
        self.assertFailsGracefullyCST('def a(')

    def test__at_def_param_type(self):
        self.assertFailsGracefullyCST('def a(t1')

    def test__at_def_param_value(self):
        self.assertFailsGracefullyCST('def a(t1 name')

    def test__at_def_param_comma(self):
        self.assertFailsGracefullyCST('def a(t1 name,')

    def test__at_def_param_second_type(self):
        self.assertFailsGracefullyCST('def a(t1 name,t2')

    def test__at_def_param_second_name(self):
        self.assertFailsGracefullyCST('def a(t1 name,t2 name2')

    def test__at_def_param_second_comma(self):
        self.assertFailsGracefullyCST('def a(t1 name,t2 name2,')

    def test__at_def_rpar_noargs(self):
        self.assertFailsGracefullyCST('def a()')

    def test__at_def_rpar_1_arg_comma(self):
        self.assertFailsGracefullyCST('def a(t1 name,)')

    def test__at_def_rpar_1_arg_no_comma(self):
        self.assertFailsGracefullyCST('def a(t1 name)')

    def test__at_def_rpar_2_arg(self):
        self.assertFailsGracefullyCST('def a(t1 name,t2 name2)')

    def test__at_def_lbrace(self):
        self.assertFailsGracefullyCST('def a(t1 name,t2 name2){')

    def test__at_def_rbrace(self):
        self.assertValidParseCST('def a(t1 name,t2 name2){}')

    def test__at_block_no_semi(self):
        self.assertFailsGracefullyCST('def a(t1 name,t2 name2){let a=1')

    def test__at_block_semi(self):
        self.assertFailsGracefullyCST('def a(t1 name,t2 name2){let a=1;')

    def test__common_flow_kwd(self):
        for name in ('while', 'repeat', 'while'):
            with self.subTest(name=name):
                self.assertFailsGracefullyCST(f'{name}')

    def test__common_flow_cond(self):
        for name in ('while', 'repeat', 'while'):
            with self.subTest(name=name):
                self.assertFailsGracefullyCST(f'{name} a+1')

    def test__common_flow_lbrace(self):
        for name in ('while', 'repeat', 'while'):
            with self.subTest(name=name):
                self.assertFailsGracefullyCST(f'{name} a + 1 {{')

    def test__if_full(self):
        self.assertValidParseCST('if a+1{}')

    def test__else_or_elseif__else(self):
        self.assertFailsGracefullyCST('if 1{}else')

    def test__elseif_else_if(self):
        self.assertFailsGracefullyCST('if 1{}else if')

    def test__elseif_expr(self):
        self.assertFailsGracefullyCST('if 1{}else if 2')

    def test__elseif_lbrace(self):
        self.assertFailsGracefullyCST('if 1{}else if 2{')

    def test__else_lbrace(self):
        self.assertFailsGracefullyCST('if 1{}else {')

    def test__call_args_lpar(self):
        self.assertFailsGracefullyCST('a(')

    def test__call_args_arg1(self):
        self.assertFailsGracefullyCST('a(a')

    def test__call_args_arg1_comma(self):
        self.assertFailsGracefullyCST('a(a,')

    def test__call_args_comma_rpar(self):
        self.assertFailsGracefullyCST('a(a,)')

    def test__call_args_rpar(self):
        self.assertFailsGracefullyCST('a(a)')

    def test__call_args_noargs(self):
        self.assertFailsGracefullyCST('a()')

    def test__string_single(self):
        self.assertFailsGracefullyCST('"a"')

    def test__string_autocat(self):
        self.assertFailsGracefullyCST('"a""b"')

    def test__number_single(self):
        self.assertFailsGracefullyCST('5')

    def test__lpar(self):
        self.assertFailsGracefullyCST('(')

    def test__lpar_expr(self):
        self.assertFailsGracefullyCST('(a')

    def test__basic_chain_ident_only(self):
        self.assertFailsGracefullyCST('a')

    def test__basic_chain_dot(self):
        self.assertFailsGracefullyCST('a.')

    def test__basic_chain_dot_attr(self):
        self.assertFailsGracefullyCST('a.b')

    def test__basic_chain_lsqb(self):
        self.assertFailsGracefullyCST('a[')

    def test__basic_chain_getitem_expr(self):
        self.assertFailsGracefullyCST('a[b')

    def test__basic_chain_rsqb_full(self):
        self.assertFailsGracefullyCST('a[b]')

    def test__basic_chain_rsqb_empty(self):
        self.assertFailsGracefullyCST('a[]')

    def test__basic_chain_call_lpar(self):
        self.assertFailsGracefullyCST('a(')

    def test__basic_chain_call_inner(self):
        self.assertFailsGracefullyCST('a(b')

    def test__basic_chain_call_rpar_full(self):
        self.assertFailsGracefullyCST('a(b)')

    def test__basic_chain_call_rpar_empty(self):
        self.assertFailsGracefullyCST('a()')

    def test__unaries_single(self):
        self.assertFailsGracefullyCST('+')

    def test__unaries_many(self):
        self.assertFailsGracefullyCST('-+--!')

    def test__unaries_many_expr(self):
        self.assertFailsGracefullyCST('-+--!a')

    def test__pow_op(self):
        self.assertFailsGracefullyCST('a**')

    def test__pow_rhs_unary(self):
        self.assertFailsGracefullyCST('a**-')

    def test__pow_rhs_unary_expr(self):
        self.assertFailsGracefullyCST('a**-a')

    def test__pow_rhs_expr(self):
        self.assertFailsGracefullyCST('a**b')

    def test__bin_op__op(self):
        for op in BINARY_OPS:
            with self.subTest(op=op):
                self.assertFailsGracefullyCST(f'a{op}')

    def test__bin_op__expr(self):
        for op in BINARY_OPS:
            with self.subTest(op=op):
                self.assertFailsGracefullyCST(f'a{op}b')

    def test__bin_op__success(self):
        for op in BINARY_OPS:
            with self.subTest(op=op):
                self.assertValidParseCST(f'a{op}b;')


if __name__ == '__main__':
    unittest.main()
