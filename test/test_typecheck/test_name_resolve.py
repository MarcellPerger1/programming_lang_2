from unittest.mock import patch

from parser.common import StrRegion
from parser.typecheck.name_resolver import NameResolver
from parser.typecheck.scope import NameInfo, FuncInfo, ParamInfo, Scope
from parser.typecheck.types import ValType, BoolType, VoidType
from test.common import CommonTestCase, MethodMock


class TestNameResolve(CommonTestCase):
    def test_top_scope_attr(self):
        src = 'let a = 8, b = 5; a += b; def c(val param) {c(param, a, b);}'
        # Use NameResolver._init as it's reliably called exactly once in slow path of .run()
        with patch.object(NameResolver, '_init', MethodMock(wraps=NameResolver._init)) as m:
            nr = self.getNameResolver(src)
            self.assertIsNone(nr.top_scope)
            m.assert_not_called()
            v = nr.run()
            self.assertIs(v, nr.top_scope)
            m.assert_called_once()
            v2 = nr.run()
            self.assertIs(v2, v)
            self.assertIs(v2, nr.top_scope)
            m.assert_called_once()  # Still only once

    def test_params(self):
        src = ('def f1(bool b0, val v0, string s0, number n0) {let L0=s0..v0;};'
               'def f2() {}')
        sc = Scope()
        f1_scope = Scope()
        f1_scope.declared = {
            'b0': NameInfo(f1_scope, 'b0', BoolType(), is_param=True),
            'v0': (v0 := NameInfo(f1_scope, 'v0', ValType(), is_param=True)),
            's0': (s0 := NameInfo(f1_scope, 's0', ValType(), is_param=True)),
            'n0': NameInfo(f1_scope, 'n0', ValType(), is_param=True),
            'L0': NameInfo(f1_scope, 'L0', ValType())
        }
        f1_scope.used = {'v0': v0, 's0': s0}
        sc.declared = {
            'f1': FuncInfo.from_param_info(sc, 'f1', [
                ParamInfo('b0', BoolType()),
                ParamInfo('v0', ValType()),
                ParamInfo('s0', ValType()),  # val == string == number for now
                ParamInfo('n0', ValType()),
            ], VoidType(), f1_scope),
            'f2': FuncInfo.from_param_info(sc, 'f2', [], VoidType(), Scope())
        }
        self.assertEqual(sc, self.getNameResolver(src).run())


class TestNameResolveErrors(CommonTestCase):
    def test_undefined_var(self):
        err = self.assertNameResolveError('foo = 9;')
        self.assertContains(err.msg, "Name 'foo' is not defined")
        self.assertErrorRegion(StrRegion(0, 3), err)

    def test_var_already_declared_once(self):
        err = self.assertNameResolveError('let foo = 9; let foo;')
        self.assertContains(err.msg, "Variable already declared")
        self.assertErrorRegion(StrRegion(17, 20), err)

    def test_var_already_declared_once_with_value(self):
        err = self.assertNameResolveError('let foo = 9; let foo = 55;')
        self.assertContains(err.msg, "Variable already declared")
        self.assertErrorRegion(StrRegion(17, 20), err)

    def test_var_already_declared_many_first(self):
        err = self.assertNameResolveError('let foo = 9; let foo, bar;')
        self.assertContains(err.msg, "Variable already declared")
        self.assertErrorRegion(StrRegion(17, 20), err)

    def test_var_already_declared_many_mid(self):
        err = self.assertNameResolveError('let bar = 9; let foo, bar, baz;')
        self.assertContains(err.msg, "Variable already declared")
        self.assertErrorRegion(StrRegion(22, 25), err)

    def test_var_already_declared_many_last(self):
        err = self.assertNameResolveError('let baz = 9; let foo, bar, baz;')
        self.assertContains(err.msg, "Variable already declared")
        self.assertErrorRegion(StrRegion(27, 30), err)

    def test_fn_already_declared(self):
        err = self.assertNameResolveError('def foo(){}; def foo(){}')
        self.assertContains(err.msg, "Function already declared")
        self.assertErrorRegion(StrRegion(17, 20), err)

    def test_unknown_param_type(self):
        err = self.assertNameResolveError('def foo(not_a_type name){};')
        self.assertContains(err.msg, "Unknown parameter type")
        self.assertErrorRegion(StrRegion(8, 18), err)

    def test_duplicate_param_name(self):
        err = self.assertNameResolveError('def foo(bool a, val a){};')
        self.assertContains(err.msg, "There is already a parameter of this name")
        self.assertErrorRegion(StrRegion(20, 21), err)

    def test_fn_resued_as_var(self):
        err = self.assertNameResolveError('def foo(){}; let foo;')
        self.assertContains(err.msg, "Variable already declared")
        self.assertErrorRegion(StrRegion(17, 20), err)

    def test_var_resued_as_fn(self):
        err = self.assertNameResolveError('let foo = 9; def foo(){/*hi*/}')
        self.assertContains(err.msg, "Function already declared")
        self.assertErrorRegion(StrRegion(17, 20), err)
