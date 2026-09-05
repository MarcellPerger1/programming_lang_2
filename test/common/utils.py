"""General testing utils that could be used in any project
(not specific to this project). Utils specific to this project should go in common.py"""
from __future__ import annotations

import os
import unittest
from collections.abc import Sized
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, Protocol, Iterable, Container, Any
from unittest.mock import Mock

from unittest.util import safe_repr

T = TypeVar('T')
T_contra = TypeVar('T_contra', contravariant=True)
T_co = TypeVar('T_co', covariant=True)
U = TypeVar('U')
U_contra = TypeVar('U_contra', contravariant=True)
SizedT = TypeVar('SizedT', bound=Sized)

if TYPE_CHECKING:
    from _typeshed import SupportsDunderLE, SupportsDunderGE

    class SupportsLeAndGe(SupportsDunderLE[T_contra],
                          SupportsDunderGE[U_contra],
                          Protocol[T_contra, U_contra]):
        pass  # All from inheritance


class TestCaseUtils(unittest.TestCase):
    # Type annotations not perfect but Python's type system is insufficiently
    #  expressive due to its lack of intersection type, inflexible
    #  declaration-site variance, and underused immature Protocol system.
    def assertBetweenIncl(self, lo: T, hi: U, value: SupportsLeAndGe[U, T],
                          msg: str | None = None):
        """Just like self.assertTrue(lo <= value <= hi), but with a nicer default message."""
        if lo <= value <= hi:
            return
        standard_msg = (f'{safe_repr(value)} is not between'
                        f' {safe_repr(lo)} and {safe_repr(hi)}')
        self.fail(self._formatMessage(msg, standard_msg))

    def assertContains(self, container: Iterable | Container, member: Any, msg=None):
        self.assertIn(member, container, msg)

    @staticmethod
    def isProperCwdSet():
        # Not demon-proof but whatever... It is, however, CI-being-stupid-proof (I hope).
        return Path('./.github/workflows').exists()

    def setProperCwd(self):
        """Sets the working directory to the project root (if it isn't set already)"""
        if self.isProperCwdSet():
            return
        self._old_cwd = os.getcwd()
        dirname = Path(__file__).parent
        os.chdir(dirname.parent.parent)
        assert self.isProperCwdSet()
        self.addCleanup(self.resetCwd)

    def resetCwd(self):
        os.chdir(self._old_cwd)

    def assertHasSingleItem(self, container: Iterable[T]) -> T:
        # Overcomplicated for better error messages and generality
        it = iter(container)
        try:
            v = next(it)
        except StopIteration:
            self.fail(f"Expected iterable to contain one item, was empty: {safe_repr(it)}")
        try:
            v2 = next(it)
        except StopIteration:
            return v  # ok, ran out of items so list is singleton
        try:
            # noinspection PyTypeChecker
            length = len(it)
        except (TypeError, NotImplementedError):
            self.fail(f"Expected iterable to contain one item, got at least "
                      f"one extra (iterable: {container}, extra: {v2})")
        self.fail(f"Expected iterable to contain one item, got {length} items"
                  f"(iterable: {container})")

    def assertAsNotNone(self, x: T | None, msg: str | None = None) -> T:
        self.assertIsNotNone(x, msg)
        return x

    def assertHasLength(self, sized: SizedT | None, n: int) -> SizedT:
        sized = self.assertAsNotNone(sized)
        self.assertEqual(n, len(sized), f"Expected {sized} to have size {n}")
        return sized

    def assertAsInstance(self, o: object, cls: type[T], msg: str | None = None) -> T:
        self.assertIsInstance(o, cls, msg)
        return o


class MethodMock(Mock):
    """Mocks a method on a class, for use with patch.object.
    For example::
        with patch.object(MyClass, 'my_method', MethodMock()) as m:
    """
    # Not-so-black magic to allow setting the method on the class while ensuring
    # that the method receives the correct `self` value (builtin unittest is
    # a bit broken in this regard, passing no self value at all)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__['__mock_args'] = args
        self.__dict__['__mock_kwargs'] = kwargs
        # We use __dict__ so that Mock doesn't intercept this property access
        self.__dict__['__inst'] = None

    def __get__(self, instance, owner=None):
        # This very naive method of finding out the proper instance works
        # because __get__ will be called again every time this is accessed
        if instance is not None:
            return _BoundDelegateMock(
                self, instance, *self.__dict__['__mock_args'],
                **self.__dict__['__mock_kwargs'])
        return _DelegateMock(self, *self.__dict__['__mock_args'],
                             **self.__dict__['__mock_kwargs'])


class _DelegateMock(Mock):
    def __new__(cls, inner, /, *args, **kwargs):
        # noinspection PyTypeChecker
        return super().__new__(cls, *args, **kwargs)  # Pycharm doesn't understand __new__

    def __init__(self, inner: Mock, /, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We cannot use property setter the first time as it will not be in
        #  __dict__ so the property will not be recognised with spec_set
        self.__dict__['__inner'] = inner

    # We use __dict__ so that Mock doesn't intercept this property access
    @property
    def __inner(self) -> Mock:
        return self.__dict__['__inner']

    @__inner.setter
    def __inner(self, value: Mock):
        self.__dict__['__inner'] = value

    def __call__(self, *args, **kwargs):
        return self.__inner(*args, **kwargs)

    def assert_called_with(self, *args, **kwargs):
        self.__inner.assert_called_with(*args, **kwargs)

    def assert_not_called(self):
        self.__inner.assert_not_called()

    def assert_called_once_with(self, *args, **kwargs):
        self.__inner.assert_called_once_with(*args, **kwargs)

    def assert_called(self):
        self.__inner.assert_called()

    def assert_called_once(self):
        self.__inner.assert_called_once()

    def reset_mock(self, visited=None, *, return_value=False, side_effect=False):
        self.__inner.reset_mock(visited, return_value=return_value, side_effect=side_effect)

    def assert_any_call(self, *args, **kwargs):
        self.__inner.assert_any_call(*args, **kwargs)

    def assert_has_calls(self, calls, any_order=False):
        self.__inner.assert_has_calls(calls, any_order)

    def mock_add_spec(self, spec, spec_set=False):
        self.__inner.mock_add_spec(spec, spec_set)

    def attach_mock(self, mock, attribute):
        self.__inner.attach_mock(mock, attribute)

    def configure_mock(self, **kwargs):
        self.__inner.configure_mock(**kwargs)


class _BoundDelegateMock(_DelegateMock):
    def __new__(cls, inner, inst, /, *args, **kwargs):
        # (Pycharm doesn't understand __new__, see PY-89087)
        # noinspection PyTypeChecker
        return super().__new__(cls, inner, *args, **kwargs)

    def __init__(self, inner: Mock, inst, /, *args, **kwargs):
        super().__init__(inner, *args, **kwargs)
        self.__dict__['__inst'] = inst

    def __call__(self, *args, **kwargs):
        return super().__call__(self.__dict__['__inst'], *args, **kwargs)
