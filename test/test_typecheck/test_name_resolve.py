from unittest.mock import Mock, patch

from parser.common import StrRegion
from parser.typecheck.typecheck import (
    NameResolver, Scope, NameInfo, BoolType, ValType, VoidType,
    FuncInfo, ParamInfo)
from test.common import CommonTestCase


class BoundMock(Mock):
    # Not-so-black magic to allow setting the method on the class while ensuring
    # that the wrapper receives the correct `self` value (builtin unittest is
    # a bit broken in this regard, as it passes no self value at all)
    def __init__(self, *args, wraps=None, **kwargs):
        if wraps:
            def wraps_2(*a, **k):  # I hope we don't have to pickle this anywhere...
                if (inst := self.__dict__['_inst']) is not None:
                    return wraps(inst, *a, **k)
                else:
                    return wraps(*a, **k)
        else:
            wraps_2 = None
        super().__init__(*args, wraps=wraps_2, **kwargs)
        self.__dict__['_inst'] = None

    def __get__(self, instance, owner=None):
        # This very naive method of finding out the proper instance works
        # because __get__ will be called again every time this is accessed
        if owner is not None:  # if accessed on instance
            self.__dict__['_inst'] = instance
        else:
            self.__dict__['_inst'] = None
        return self


class TestNameResolve(CommonTestCase):
    def test_top_scope_attr(self):
        src = 'let a = 8, b = 5; a += b; def c(val param) {c(param, a, b);}'
        orig = NameResolver._init  # Reliably called exactly once in slow path of .run()
        m = BoundMock(spec_set=orig, wraps=orig)
        with patch.object(NameResolver, '_init', new_callable=lambda: m):
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
        }
        f1_scope.used = {'v0': v0, 's0': s0}
        sc.declared = {
            'f1': FuncInfo.from_param_info(sc, 'f1', [
                ParamInfo('b0', BoolType()),
                ParamInfo('v0', ValType()),
                ParamInfo('s0', ValType()),  # val == string  ==  number for now
                ParamInfo('n0', ValType()),
            ], VoidType(), f1_scope),
            'f2': FuncInfo.from_param_info(sc, 'f2', [], VoidType(), Scope())
        }
        self.assertEqual(self.getNameResolver(src).run(), sc)


class TestNameResolveErrors(CommonTestCase):
    def test_undefined_var(self):
        err = self.assertNameResolveError('foo = 9;')
        self.assertContains(err.msg, "Name 'foo' is not defined")
        self.assertEqual(StrRegion(0, 3), err.region)

    def test_var_already_declared(self):
        err = self.assertNameResolveError('let foo = 9; let foo;')
        self.assertContains(err.msg, "Variable already declared")
        self.assertEqual(StrRegion(13, 20), err.region)

    def test_fn_already_declared(self):
        err = self.assertNameResolveError('def foo(){}; def foo(){}')
        self.assertContains(err.msg, "Function already declared")
        self.assertEqual(StrRegion(17, 20), err.region)

    def test_unknown_param_type(self):
        err = self.assertNameResolveError('def foo(not_a_type name){};')
        self.assertContains(err.msg, "Unknown parameter type")
        self.assertEqual(StrRegion(8, 18), err.region)

    def test_duplicate_param_name(self):
        err = self.assertNameResolveError('def foo(bool a, val a){};')
        self.assertContains(err.msg, "There is already a parameter of this name")
        self.assertEqual(StrRegion(20, 21), err.region)
