from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import overload, TYPE_CHECKING, TypeVar, Protocol

from unittest.util import safe_repr

T = TypeVar('T')
T_contra = TypeVar('T_contra', contravariant=True)
U = TypeVar('U')
U_contra = TypeVar('U_contra', contravariant=True)

if TYPE_CHECKING:
    from _typeshed import SupportsDunderLE, SupportsDunderGE

    class SupportsLeAndGe(SupportsDunderLE[T_contra],
                          SupportsDunderGE[U_contra],
                          Protocol[T_contra, U_contra]):
        pass  # All from inheritance

    class ExtendsAndGE(T, SupportsDunderGE[U], Protocol[T, U]):
        ...

    class ExtendsAndLE(T, SupportsDunderLE[U], Protocol[T, U]):
        ...


class TestCaseUtils(unittest.TestCase):
    @overload  # b:GE & b:LE
    def assertBetweenIncl(self, lo: T, hi: U, value: SupportsLeAndGe[U, T],
                          msg: str | None = None): ...

    @overload  # a:LE & c:GE
    def assertBetweenIncl(self, lo: SupportsDunderLE[T], hi: SupportsDunderGE[T], value: T,
                          msg: str | None = None): ...

    @overload  # b:GE & c:GE  # Does this work? Who knows?
    def assertBetweenIncl(self, lo: T, hi: SupportsDunderGE[U], value: ExtendsAndGE[U, T],
                          msg: str | None = None): ...

    @overload  # a:LE & b:LE  # Does this work? Who knows?
    def assertBetweenIncl(self, lo: SupportsDunderLE[T], hi: U, value: ExtendsAndLE[T, U],
                          msg: str | None = None): ...

    def assertBetweenIncl(self, lo: SupportsDunderLE, hi: SupportsLeAndGe,
                          value: SupportsDunderGE, msg=None):
        """Just like self.assertTrue(lo <= value <= hi), but with a nicer default message."""
        if lo <= value <= hi:
            return
        standard_msg = (f'{safe_repr(value)} is not between'
                        f' {safe_repr(lo)} and {safe_repr(hi)}')
        self.fail(self._formatMessage(msg, standard_msg))

    @staticmethod
    def isProperCwdSet():
        return Path('./.github/workflows').exists()

    def setProperCwd(self):
        """Sets the working directory to the project root (if it isn't set already)"""
        if self.isProperCwdSet():
            return
        self._old_cwd = os.getcwd()
        dirname = Path(__file__).parent
        os.chdir(dirname.parent)
        assert self.isProperCwdSet()
        self.addCleanup(self.resetCwd)

    def resetCwd(self):
        os.chdir(self._old_cwd)
