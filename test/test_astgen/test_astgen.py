from parser.astgen.ast_nodes import AstProgramNode, AstIf, AstNumber
from parser.common import StrRegion
from test.common import CommonTestCase


class TestAstGen(CommonTestCase):
    def test_op_node(self):
        self.assertAstMatchesSnapshot('s=s+9;')

    def test_nop_node(self):
        self.assertAstMatchesSnapshot(
            'if(2){;}else if (1==2){;a();;} else {};\n'
            ';repeat(a){;};def fn(){;};')

    def test_while_block(self):
        self.assertAstMatchesSnapshot(
            'while -1==1+-2 {};\n'
            'while (Abc.d.lower()==msg[0].lower()){\n'
            '  print("q" .. msg[0]);\n'
            '}')

    def test_aug_assign(self):
        self.assertAstMatchesSnapshot(
            'a += 6 + !0;\n'
            'a.b[c .. "w"] **= (-0.5).q();'
            'a.c ||= "\\t\\x1b[34;45m(\\"default\\")\\x1b[0m\\n";')

    def test_error(self):
        err = self.assertFailsGracefullyAST('3 + a;')
        self.assertEqual(StrRegion(0, 5), err.region)
        self.assertContains(err.msg.lower(), "expected statement")

        err = self.assertFailsGracefullyAST('b*3 = 8;')
        self.assertEqual(StrRegion(0, 3), err.region)
        self.assertContains(err.msg.lower(), "cannot assign")

    def test_getattr(self):
        self.assertAstMatchesSnapshot('a.b[c].d = e.f[g].h[i];')

    def test_string_basic(self):  # Just test very basic string stuff, rest is in eval_string
        self.assertAstMatchesSnapshot('a="a\\ueDf9\\t";'
                                      "b = 'q\\a\\'q';")

    def test_autocat(self):
        self.assertAstMatchesSnapshot(
            'b="abc\\U0010f9aB"  ' + "'end1234'\n" + "'\"'" + '"\'";')

    def test_unaries(self):
        self.assertAstMatchesSnapshot('a=+(-!b==!-+c)-+--r+(-9);')

    def test_decl(self):
        self.assertAstMatchesSnapshot('let a,b=1+1,c;\n'
                                      'global d = "STRING", e;\n'
                                      'let[] local_list=list(), other;\n'
                                      'global[] STACK;')

    def test_list_literal_decl(self):
        self.assertAstMatchesSnapshot('let[] loc = [5, 6.,], b, c=[];\n'
                                      'global [] STACK = [foo(bar), 8];')

    def test_list_literal_decl_paren(self):
        self.assertAstMatchesSnapshot('let[] a = ([1]);')
    # We only test for lists in variable decls (as them being allowed
    # elsewhere is UB for now).

    def test_elif_chain(self):
        src = "if 1{} else if 6{} else if 7{} else{}"
        a = self.assertAsInstance(self.getAstGen(src).parse(), AstProgramNode)
        if_0 = self.assertAsInstance(self.assertHasSingleItem(a.statements), AstIf)
        self.assertEqual(AstNumber(StrRegion(3, 4), 1), if_0.cond)
        self.assertEqual(if_0.if_body, [])
        self.assertIsInstance(if_0.else_body, list)
        elif_1 = self.assertAsInstance(
            self.assertHasSingleItem(self.assertAsNotNone(if_0.else_body)),
            AstIf)
        self.assertEqual(AstNumber(StrRegion(15, 16), 6), elif_1.cond)
        self.assertEqual(elif_1.if_body, [])
        self.assertIsInstance(elif_1.else_body, list)
        elif_2 = self.assertAsInstance(
            self.assertHasSingleItem(self.assertAsNotNone(elif_1.else_body)),
            AstIf)
        self.assertEqual(AstNumber(StrRegion(28, 28), 7), elif_2.cond)
        self.assertEqual(elif_2.if_body, [])
        self.assertEqual(elif_2.else_body, [])
        self.assertMatchesSnapshot(a)  # In case I missed any checks
