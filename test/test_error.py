import unittest
from unittest.mock import patch

from parser.common import StrRegion
from parser.common.error import BaseLocatedError


class TestLocatedError(unittest.TestCase):
    def test_adds_note_once(self):
        with patch.object(BaseLocatedError, 'display_region') as mock_display:
            mock_display.return_value = 'mock_return_value'
            err = BaseLocatedError("a_message", StrRegion(0, 2), 'src_str')
            str(err)
            self.assertEqual(len(err.get_notes()), 1)
            str(err)
            self.assertEqual(len(err.get_notes()), 1)

    def test_uses_display_return_value(self):
        with patch.object(BaseLocatedError, 'display_region') as mock_display:
            mock_display.return_value = 'mock_return_value'
            err = BaseLocatedError("a_message", StrRegion(0, 2), 'src_str')
            str(err)
            mock_display.assert_called_once_with('src_str', StrRegion(0, 2))
            self.assertEqual(err.get_notes(), ['mock_return_value'])

    def test_only_calls_display_when_needed(self):
        with patch.object(BaseLocatedError, 'display_region') as mock_display:
            mock_display.return_value = 'mock_return_value'
            err = BaseLocatedError("a_message", StrRegion(0, 2), 'src_str')
            mock_display.assert_not_called()
            str(err)
            mock_display.assert_called_once()
            str(err)
            mock_display.assert_called_once()  # still only once


if __name__ == '__main__':
    unittest.main()
