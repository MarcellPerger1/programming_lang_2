import os
import unittest
from pathlib import Path

from main import main


class TestMain(unittest.TestCase):
    def setUp(self):
        if not Path('./main_example_0.st').exists():
            self.old_dir = os.getcwd()
            os.chdir(Path(__file__).parent.parent)
            self.addCleanup(lambda : os.chdir(self.old_dir))

    def test(self):
        main()  # TODO: add some sort of snapshot for main?


if __name__ == '__main__':
    unittest.main()
