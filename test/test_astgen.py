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
