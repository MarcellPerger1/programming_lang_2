from parser.astgen.ast_nodes import AstDeclNode
from parser.typecheck.typecheck import TypeMetadata
from parser.typecheck.types import ValType, VoidType
from test.common import CommonTestCase


class TestGivenTypes(CommonTestCase):
    def test_var_decl_assign(self):
        prog = self.getTypechecker("let a = 9;").run()
        self.assertAllMetadata(prog, TypeMetadata)
        self.assertTypecheckedTo(prog, VoidType())
        decl = self.assertHasSingleItem(prog.statements)
        self.assertTypecheckedTo(decl, VoidType())
        self.assertIsInstance(decl, AstDeclNode)
        decl: AstDeclNode[TypeMetadata]
        self.assertTypecheckedTo(decl.value, ValType())
        self.assertTypecheckedTo(decl.ident, ValType())
        # self.assertMatchesSnapshot(prog)

    def test_function(self):
        prog = self.getTypechecker("def f(val a, bool b, number c, string d){}").run()
        self.assertAllMetadata(prog, TypeMetadata)
        # self.assertMatchesSnapshot(prog)
