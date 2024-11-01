from unittest import TestCase

from parser.common.str_region import StrRegion
from parser.tokens import NumberToken, IdentNameToken
from parser.cst.named_node import node_from_token
from parser.cst.nodes import NumberNode, IdentNode


class Test(TestCase):
    def test_node_from_token(self):
        self.assertIsInstance(node_from_token(NumberToken()), NumberNode)
        self.assertEqual(NumberNode(None, None), node_from_token(NumberToken()))
        self.assertIsInstance(node_from_token(IdentNameToken()), IdentNode)
        self.assertEqual(IdentNode(None, None), node_from_token(IdentNameToken()))
        self.assertEqual(IdentNode(StrRegion(5, 7), None),
                         node_from_token(IdentNameToken(StrRegion(5, 7))))

