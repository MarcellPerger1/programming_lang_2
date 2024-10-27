import unittest

from parser.lexer import Tokenizer
from parser.tokens import *
from test.common import CommonTestCase, TokenStreamFlag


class MyTestCase(CommonTestCase):
    def test_ws_at_end(self):
        t = Tokenizer('let a =1; \n').tokenize()
        self.assertTokensEqual(t, [
            IdentNameToken(),
            WhitespaceToken(),
            IdentNameToken(),
            WhitespaceToken(),
            OpToken(op_str='='),
            NumberToken(),
            SemicolonToken(),
            WhitespaceToken(),
            EofToken()
        ], TokenStreamFlag.FULL, check_regions=False)


if __name__ == '__main__':
    unittest.main()
