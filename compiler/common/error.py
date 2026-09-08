from __future__ import annotations

import traceback

from .common import HasRegion
from .str_region import StrRegion

__all__ = ['BaseParseError', 'BaseLocatedError']


# We unconditionally add this polyfill to ensure that
# exceptions are always part of str(exc).
# Note that the 'really clever' way of implementing this
# (@property __notes__) FAILS because traceback also uses
# __notes__ to print the extra non-str() stuff on the end.
# So we cannot distinguish between whether it was called
# by add_note() (and we should return a ref to __polyfill_notes__)
# or by print_exc() (and we should return [] so notes are only added once)
# (Well except by looking at stack frames and similar hackery
#  but even I'm not willing to go that far!
#  (and it might fail when called from C extension))
# This means that the rest of the code will just have to deal with it
# and use get_notes() instead of .__notes__
class _PolyfillAddNoteMixin:
    """Usage: class MyError(_PolyfillAddNoteMixin, Exception): ...
    (**The order is important!**)"""
    # Don't use actual attribute because then we would include it in str()
    # and notes would also be printed below by traceback.print_exc (so twice)
    __polyfill_notes__: list[str]

    # We are first in the MRO so we override Exception's stuff for notes
    def add_note(self, note: str, /):
        self._transfer_builtin_notes()
        self._add_note(note)

    def get_notes(self):
        return getattr(self, '__polyfill_notes__', [])

    def _add_note(self, note: str, /):
        if not hasattr(self, '__polyfill_notes__'):
            self.__polyfill_notes__ = []
        self.__polyfill_notes__.append(note)

    # It is possible that we are actually not the first in MRO
    # (because of incorrect usage?) so we provide this method
    # to transfer the notes to our polyfill
    def _transfer_builtin_notes(self):
        for note in getattr(self, '__notes__', None) or []:
            self._add_note(str(note))
        try:
            # noinspection PyUnresolvedReferences
            del self.__notes__  # Clear added
        except AttributeError:
            pass

    def __str__(self):
        self._transfer_builtin_notes()
        # here super() refers to the next thing in MRO,
        # meaning the class after this in the bases
        if notes := getattr(self, '__polyfill_notes__', []):
            return super().__str__() + '\n' + '\n'.join(notes)
        return super().__str__()


class BaseParseError(_PolyfillAddNoteMixin, Exception):
    pass


class BaseLocatedError(BaseParseError, HasRegion):
    def __init__(self, msg: str, region: StrRegion, src: str):
        super().__init__(msg)
        self.msg = msg
        self._src_text = src
        self.region = region
        self._has_added_note = False

    # Don't need to send over traceback (for now) as Python prints
    # all the errors from other processes as well .
    # (But it would be a nice challenge for when I have time)
    # (Then again so would a lot of other projects I want to do)
    def __getnewargs__(self):
        # This function is needed here because pickle is a bit stupid
        # (dunno why?) so need to tell it the new args.
        return self.msg, self.region, self._src_text

    def compute_location(self):
        if self._has_added_note:
            return
        # noinspection PyBroadException
        try:
            self.add_note(self.region.display(self._src_text))
        except Exception:
            short_loc = f'{self.region.start} to {self.region.end - 1}'
            self.add_note(
                '\nAn error occurred while trying to display the location '
                f'of the ParseError ({short_loc}):\n{traceback.format_exc()}')
        self._has_added_note = True

    def __str__(self):
        self.compute_location()
        return super().__str__()
