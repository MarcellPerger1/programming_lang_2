from test.common import CommonTestCase


class TestAstGen(CommonTestCase):
    def test_op_node(self):
        self.assertAstMatchesSnapshot('s=s+9;')
