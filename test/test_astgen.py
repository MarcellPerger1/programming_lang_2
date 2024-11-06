from parser.astgen.astgen import AstGen
from parser.cst.treegen import CstGen
from parser.lexer.tokenizer import Tokenizer
from test.common import CommonTestCase


class TestAstGen(CommonTestCase):
    def assertAstParses(self, src: str):
        a = AstGen(CstGen(Tokenizer(src)))
        self.assertIsNotNone(a.walk())

    def test_op_node(self):
        self.assertAstParses('s=s+9;')
