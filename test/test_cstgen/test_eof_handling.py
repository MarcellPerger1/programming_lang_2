from parser.operators import BINARY_OPS
from test.common import CommonTestCase


class TestCstGenEof(CommonTestCase):
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
