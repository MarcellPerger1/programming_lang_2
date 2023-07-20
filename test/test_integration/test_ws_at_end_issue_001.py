import unittest
from pathlib import Path

from parser.tokenizer import Tokenizer
from parser.tokens import *

filepath = Path(__file__)


def load_example():
    path = (filepath.parent / 'examples'
            / filepath.with_suffix('.txt').name.removeprefix('test_'))
    with open(path) as f:
        return f.read()


class MyTestCase(unittest.TestCase):
    def _token_as_tuple(self, t: Token):
        if isinstance(t, OpToken):
            return t.name, t.op_str
        return (t.name, )

    def assert_tokens_match(self, a: list[Token], b: list[Token]):
        a_tuples = map(self._token_as_tuple, a)
        b_tuples = map(self._token_as_tuple, b)
        self.assertEqual(a, b)

    def test_ws_at_end(self):
        t = Tokenizer(load_example()).tokenize()
        self.assert_tokens_match(t.tokens, [
            IdentNameToken(),
            WhitespaceToken(),
            IdentNameToken(),
            OpToken(op_str='='),
            NumberToken(),
            SemicolonToken(),
            WhitespaceToken()
        ])


if __name__ == '__main__':
    unittest.main()
