from parser.astgen.ast_node import AstNode
from parser.astgen.ast_nodes import AstDeclNode, AstDefine, AstAugAssign, AstWhile, \
    VarDeclType, VarDeclScope, AstRepeat, AstItem
from parser.common import StrRegion
from parser.typecheck.types import ValType, VoidType, BoolType, TypeType, TypeMetadata, \
    ListType
from test.common import CommonTestCase


# We don't do snapshot tests as those would only test the astgen (as we manually
#  check for all the type info). Less maintenacne burden as well.
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

    def test_function(self):
        prog = self.getTypechecker("def f(val a, bool b, number c, string d){}").run()
        self.assertAllMetadata(prog, TypeMetadata)
        f = self.assertAsInstance(self.assertHasSingleItem(prog.statements), AstDefine)
        expected_types = [ValType(), BoolType(), ValType(), ValType()]
        self.assertEqual(len(expected_types), len(f.params))
        for expect_t, param in zip(expected_types, f.params):
            self.assertTypecheckedTo(param, VoidType())
            self.assertTypecheckedTo(param.ident, expect_t)
            self.assertTypecheckedTo(param.type, TypeType(expect_t))

    def test_operators(self):
        prog = self.getTypechecker("let a = (1 + 1) * 2.2;").run()
        self.assertAllMetadata(prog, TypeMetadata)

    def test_bools(self):
        prog = self.getTypechecker("if(!(6==7) && (8==9 || 7<2) || 4>=2){}").run()
        self.assertAllMetadata(prog, TypeMetadata)

    def test_assign_list_getitem(self):
        prog = self.getTypechecker("global AA = 9; let[] bb = [7]; AA = bb[1];").run()
        self.assertAllMetadata(prog, TypeMetadata)

    def test_aug_assign(self):
        prog = self.getTypechecker("let v=8; v+=5;").run()
        self.assertAllMetadata(prog, TypeMetadata)
        decl, aug = self.assertHasLength(prog.statements, 2)
        self.assertTypecheckedTo(decl, VoidType())
        self.assertTypecheckedTo(aug, VoidType())
        self._check_decl_val(decl)
        aug = self.assertAsInstance(aug, AstAugAssign)
        self.assertTypecheckedTo(aug.target, ValType())
        self.assertTypecheckedTo(aug.source, ValType())

    def test_while(self):
        prog = self.getTypechecker("while(1==1){let[] b;}").run()
        self.assertAllMetadata(prog, TypeMetadata)
        while_ = self.assertAsInstance(self.assertHasSingleItem(prog.statements), AstWhile)
        self.assertTypecheckedTo(while_, VoidType())
        self.assertTypecheckedTo(while_.cond, BoolType())
        let = self.assertAsInstance(self.assertHasSingleItem(while_.body), AstDeclNode)
        self.assertTypecheckedTo(let, VoidType())
        self.assertTypecheckedTo(let.ident, ListType())
        self.assertIsNone(let.value)
        self.assertEqual(VarDeclType.LIST, let.type)
        self.assertEqual(VarDeclScope.LET, let.scope)

    def test_repeat(self):
        prog = self.getTypechecker("repeat 42 {}").run()
        self.assertAllMetadata(prog, TypeMetadata)
        repeat = self.assertAsInstance(self.assertHasSingleItem(prog.statements), AstRepeat)
        self.assertTypecheckedTo(repeat, VoidType())
        self.assertTypecheckedTo(repeat.count, ValType())
        self.assertHasLength(repeat.body, 0)

    def test_string_getitem(self):
        prog = self.getTypechecker("let a='hello'; let b=a[4];").run()
        self.assertAllMetadata(prog, TypeMetadata)
        a, b = self.assertHasLength(prog.statements, 2)
        self.assertTypecheckedTo(a, VoidType())
        self.assertTypecheckedTo(b, VoidType())
        self._check_decl_val(a)
        b = self._check_decl_val(b)
        getitem = self.assertAsInstance(b.value, AstItem)
        self.assertTypecheckedTo(getitem, ValType())
        self.assertTypecheckedTo(getitem.obj, ValType())
        self.assertTypecheckedTo(getitem.index, ValType())

    def test_no_value_let(self):
        prog = self.getTypechecker("let a;").run()
        self.assertAllMetadata(prog, TypeMetadata)
        let = self.assertAsInstance(self.assertHasSingleItem(prog.statements), AstDeclNode)
        self.assertTypecheckedTo(let, VoidType())
        self.assertTypecheckedTo(let.ident, ValType())
        self.assertIsNone(let.value)

    def _check_decl_val(self, a: AstNode):
        a = self.assertAsInstance(a, AstDeclNode)
        self.assertTypecheckedTo(a.ident, ValType())
        self.assertTypecheckedTo(a.value, ValType())
        return a


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
        # BEHAV: Or 11->18 (either including or excluding parens? - which one?)
        self.assertErrorRegion(StrRegion(12, 17), exc)

    def test_bool_only_in_condition(self):
        exc = self.assertTypecheckError("let a=8;\nif a {}")
        self.assertEqual(exc.msg, "Expected type bool, got type val")
        self.assertErrorRegion(StrRegion(12, 13), exc)

    def test_assign_func(self):
        exc = self.assertTypecheckError("def f(){}\nf = 8;")
        self.assertErrorRegion(StrRegion(10, 11), exc)
        self.assertEqual(exc.msg, "Cannot assign directly to () -> void")

    def test_call_extra_arg_single(self):
        exc = self.assertTypecheckError("def f(){}; f(6769);")
        self.assertErrorRegion(StrRegion(13, 17), exc)
        self.assertEqual("Incorrect number of arguments, expected 0, got 1", exc.msg)

    def test_call_error_one_extra_arg(self):
        exc = self.assertTypecheckError("def f(val a){}; f(67, 69);")
        self.assertErrorRegion(StrRegion(22, 24), exc)
        self.assertEqual("Incorrect number of arguments, expected 1, got 2", exc.msg)

    def test_call_error_many_extra_arg(self):
        # BEHAV: should we do last or first extraneous one? Do first for
        #  now as last might imply that it's the only extraneous arg. Or all??
        exc = self.assertTypecheckError("def f(){}; f(67, 69);")
        self.assertErrorRegion(StrRegion(13, 15), exc)
        self.assertEqual("Incorrect number of arguments, expected 0, got 2", exc.msg)

    def test_call_not_enough_arg(self):
        # BEHAV: Should we really highlight entire call? Fine I guess for now
        exc = self.assertTypecheckError("def f(val s, val r){}; f(67);")
        self.assertErrorRegion(StrRegion(23, 28), exc)
        self.assertEqual("Incorrect number of arguments, expected 2, got 1", exc.msg)

    def test_call_not_enough_arg_given_zero(self):
        exc = self.assertTypecheckError("def f(val s, val r){}; f();")
        self.assertErrorRegion(StrRegion(23, 26), exc)
        self.assertEqual("Incorrect number of arguments, expected 2, got 0", exc.msg)

    def test_call_list(self):
        exc = self.assertTypecheckError("let[] a=[6]; a();")
        self.assertEqual("Cannot call list", exc.msg)
        self.assertErrorRegion(StrRegion(13, 14), exc)

    def test_call_val(self):
        exc = self.assertTypecheckError("let a=8; a();")
        self.assertEqual("Cannot call val", exc.msg)
        self.assertErrorRegion(StrRegion(9, 10), exc)

    def test_item_of_func(self):
        exc = self.assertTypecheckError("def f(val x){}; let a=f[9];")
        self.assertEqual("Cannot get item of (val) -> void", exc.msg)
        self.assertErrorRegion(StrRegion(22, 26), exc)

    def test_attr_doesnt_crash(self):
        with self.assertDoesntCrash():
            self.getTypechecker("let a; a.b=9;").run()
        with self.assertDoesntCrash():
            self.getTypechecker("let a; let c = a.b;").run()

    def test_unsupported_aug_assign_doesnt_crash(self):
        with self.assertDoesntCrash():
            self.getTypechecker("let a; a-=7;").run()
        with self.assertDoesntCrash():
            self.getTypechecker("let a; a**=7;").run()
        with self.assertDoesntCrash():
            self.getTypechecker("let[] a; a[6] += 7;").run()
