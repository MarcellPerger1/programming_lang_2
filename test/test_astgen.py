from test.common import CommonTestCase


class TestAstGen(CommonTestCase):
    def test_op_node(self):
        self.assertValidParseAST('s=s+9;')
