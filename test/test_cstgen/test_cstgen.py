import unittest

from parser.lexer.tokenizer import Tokenizer
from parser.cst.cstgen import CstGen, LocatedCstError
from parser.common import StrRegion
from test.common import CommonTestCase


# region ---- <Test Expressions> ----
class TestExpr(CommonTestCase):
    def test_autocat(self):
        self.assertCstMatchesSnapshot('"abc" "="\n  "1" .. a .. "b".d();')

    def test_mod_supported(self):
        self.assertCstMatchesSnapshot('c=a%b;')

    def test_pow(self):
        self.assertCstMatchesSnapshot('a**b;')


class TestItemChain(CommonTestCase):
    def test_item_chain(self):
        self.assertCstMatchesSnapshot('a[7].b.0.fn["c" .. 2] = fn(9).k[7 + r](3,);')

    def test_fn_call_in_lvalue(self):
        self.assertCstMatchesSnapshot('a(7).b.0.fn()["c" .. 2] = fn(9).k[7 + r](3,);')

    def test_empty_sqb_error(self):
        with self.assertRaises(LocatedCstError) as err:
            CstGen(Tokenizer('v=a[]+b;')).parse()
        exc = err.exception
        self.assertBetweenIncl(3, 4, exc.region.start)
        self.assertEqual(4, exc.region.end - 1)

    def test_empty_getitem_comma_error(self):
        exc = self.assertFailsGracefullyCST('v=a[1,]+b;')
        self.assertEqual(StrRegion(5, 6), exc.region)

    def test_getattr__issue_09(self):
        t = Tokenizer('fn(call_arg).a;').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node, 'after_call')
        t = Tokenizer('(paren + x).b;').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node, 'after_paren')
        t = Tokenizer('"a string".b;').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node, 'after_string')

    def test_getitem__issue_09(self):
        t = Tokenizer('fn(call_arg)[1];').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node, 'after_call')
        t = Tokenizer('(paren + x)[2];').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node, 'after_paren')
        t = Tokenizer('"a string"["key_" .. 3];').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node, 'after_string')

    def test_list_literal_empty(self):
        self.assertCstMatchesSnapshot('let a = [];\n'
                                      'join([], a, [].length);')
        self.assertFailsGracefullyCST('let b = [,];')

    def test_list_literal_single_item(self):
        self.assertCstMatchesSnapshot('["a"].method([-2**--6,]);')

    def test_list_literal_multi_item(self):
        self.assertCstMatchesSnapshot(
            'global[] STACK = ["int", 0];\n'
            '(STACK + ["float", .2,]).extend([6.6, 1, fn()[0], 2][-1]);\n'
            'v+=[1,2](3);')
        e = self.assertFailsGracefullyCST('[1;')
        self.assertEqual(StrRegion(2, 3), e.region)
        self.assertContains(e.msg.lower(), "after item in list")
# endregion </Test Expressions>


# region ---- <Test Statements> ----
class TestSmt(CommonTestCase):
    def test_aug_assign(self):
        self.assertCstMatchesSnapshot('a[1] += a.2;')

    def test_empty_smt__issue_04(self):
        t = Tokenizer('let a=9;;').tokenize()
        node = CstGen(t).parse()
        self.assertMatchesSnapshot(node)


class TestDecl(CommonTestCase):
    def test_empty_assign_source_error(self):
        t = Tokenizer('let a= ;').tokenize()
        with self.assertRaises(LocatedCstError) as err:
            CstGen(t).parse()
        self.assertBetweenIncl(5, 7, err.exception.region.start)
        self.assertBetweenIncl(7, 8, err.exception.region.end)
        self.assertContains(str(err.exception), "semi")

    def test_decl_no_value(self):
        self.assertCstMatchesSnapshot('let b;')

    def test_decl_multiple(self):
        self.assertCstMatchesSnapshot(
            'let a, b=9, c,d=w.1[2],e="w",f,g;\n'
            'global z,y=-11, x , w="a".lower() ,v=o , u, t;')
    
    def test_decl(self):
        self.assertCstMatchesSnapshot('let a,b=1+1,c;\n'
                                      'global d = "STRING", e;\n'
                                      'let[] local_list=list(), other;\n'
                                      'global[] STACK;')

    def test_decl_error(self):
        err = self.assertFailsGracefullyCST('let[4] = 9;')
        self.assertEqual(StrRegion(4, 5), err.region)
        self.assertContains(err.msg.lower(), "expected ']' after '['")

        err = self.assertFailsGracefullyCST('let[')
        self.assertEqual(StrRegion(4, 4), err.region)
        self.assertContains(err.msg.lower(), "expected ']' after '['")


class TestBlocks(CommonTestCase):
    def test_while(self):
        self.assertCstMatchesSnapshot('while a || !b && c >= 6 {}')
        self.assertCstMatchesSnapshot('while!(7%8){(7.7).abc(6,7,8);}')

    def test_repeat(self):
        self.assertCstMatchesSnapshot('repeat a || !b && c >= 6 {}')
        self.assertCstMatchesSnapshot('repeat!(7%8){(7.7).abc(6,7,8);}')

    def test_empty_condition_error(self):
        t = Tokenizer('if {x();}').tokenize()
        with self.assertRaises(LocatedCstError) as err:
            CstGen(t).parse()
        self.assertBetweenIncl(0, 3, err.exception.region.start)
        self.assertBetweenIncl(2, 4, err.exception.region.end)
        self.assertContains(str(err.exception), "brace")

    def test_else_if_else(self):
        self.assertCstMatchesSnapshot('if(1){}else if(a||!b&&c!=6){}')
        self.assertCstMatchesSnapshot('if(1){}else{a();}')
        self.assertCstMatchesSnapshot('if(1){}else if 9{a();}else{b(a, a());}')

    def test_else_cond_null(self):
        src = 'if 0==1{exit();  } startup();'
        n = CstGen(Tokenizer(src)).parse()
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
# endregion </Test Statements>


if __name__ == '__main__':
    unittest.main()
