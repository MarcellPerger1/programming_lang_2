from test.common import CommonTestCase


class TestAstGen(CommonTestCase):
    def test_op_node(self):
        self.assertAstMatchesSnapshot('s=s+9;')

    def test_nop_node(self):
        self.assertAstMatchesSnapshot('if(2){;}else if (1==2){;a();;} else {};\n'
                                      ';repeat(a){;};def fn(){;};')
