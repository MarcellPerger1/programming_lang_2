from parser.astgen.ast_nodes import AstDeclNode, AstDefine
from parser.common import StrRegion
from parser.typecheck.types import ValType, VoidType, BoolType, TypeType, TypeMetadata
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
        f = self.assertAsInstance(self.assertHasSingleItem(prog.statements), AstDefine)
        expected_types = [ValType(), BoolType(), ValType(), ValType()]
        self.assertEqual(len(expected_types), len(f.params))
        for expect_t, (type_ident, name_ident) in zip(expected_types, f.params):
            self.assertTypecheckedTo(name_ident, expect_t)
            self.assertTypecheckedTo(type_ident, TypeType(expect_t))
        # self.assertMatchesSnapshot(prog)

    def test_operators(self):
        prog = self.getTypechecker("let a = (1 + 1) * 2.2;").run()
        self.assertAllMetadata(prog, TypeMetadata)

    def test_bools(self):
        prog = self.getTypechecker("if(!(6==7) && (8==9 || 7<2) || 4>=2){}").run()
        self.assertAllMetadata(prog, TypeMetadata)

    def test_assign_list_getitem(self):
        prog = self.getTypechecker("global AA = 9; let[] bb = [7]; AA = bb[1];").run()
        self.assertAllMetadata(prog, TypeMetadata)


class TestErrors(CommonTestCase):
    def test_cant_pass_list_to_val_param(self):
        exc = self.assertTypecheckError("def f(val v){}\nglobal[] L=[];\nf(L);")
        self.assertEqual(exc.msg, "Expected type val, got type list")
        self.assertErrorRegion(StrRegion(32, 33), exc)
        exc = self.assertTypecheckError("def f(val v){}\nglobal[] L=[];\ndef g(){f(L);}")
        self.assertEqual(exc.msg, "Expected type val, got type list")
        self.assertErrorRegion(StrRegion(40, 41), exc)

    def test_assignment_error(self):
        exc = self.assertTypecheckError("let a; a = (1 < 2);")
        self.assertEqual(exc.msg, "Expected type val, got type bool")
        # Or 11->18 (either including or excluding parens? - which one?)
        self.assertErrorRegion(StrRegion(12, 17), exc)

    def test_bool_only_in_condition(self):
        exc = self.assertTypecheckError("let a=8;\nif a {}")
        self.assertEqual(exc.msg, "Expected type bool, got type val")
        self.assertErrorRegion(StrRegion(12, 13), exc)
