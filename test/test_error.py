import unittest
from unittest.mock import patch, MagicMock

from parser.common import StrRegion
from parser.common.error import BaseLocatedError


class _PatchRegionDisplay:
    def __init__(self):
        def _mocked_display(*args, **kwargs):
            return self._mock(*args, **kwargs)
        self._mock = MagicMock()
        self._inner = patch.object(StrRegion, 'display', _mocked_display)

    def __enter__(self):
        self._inner.__enter__()
        return self._mock

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._inner.__exit__(exc_type, exc_val, exc_tb)


class TestLocatedError(unittest.TestCase):
    def test_adds_note_once(self):
        with _PatchRegionDisplay() as mock_display:
            mock_display.return_value = 'mock_return_value'
            err = BaseLocatedError("a_message", StrRegion(0, 2), 'src_str')
            str(err)
            self.assertEqual(len(err.get_notes()), 1)
            str(err)
            self.assertEqual(len(err.get_notes()), 1)

    def test_uses_display_return_value(self):
        with _PatchRegionDisplay() as mock_display:
            mock_display.return_value = 'mock_return_value'
            err = BaseLocatedError("a_message", StrRegion(0, 2), 'src_str')
            str(err)
            mock_display.assert_called_once_with(StrRegion(0, 2), 'src_str')
            self.assertEqual(err.get_notes(), ['mock_return_value'])

    def test_only_calls_display_when_needed(self):
        with _PatchRegionDisplay() as mock_display:
            mock_display.return_value = 'mock_return_value'
            err = BaseLocatedError("a_message", StrRegion(0, 2), 'src_str')
            mock_display.assert_not_called()
            str(err)
            mock_display.assert_called_once()
            str(err)
            mock_display.assert_called_once()  # still only once


if __name__ == '__main__':
    unittest.main()
