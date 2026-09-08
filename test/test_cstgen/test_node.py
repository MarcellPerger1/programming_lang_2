from unittest import TestCase

from parser.common import StrRegion
from parser.cst.base_node import Node, Leaf, node_from_token
from parser.cst.nodes import NumberNode, IdentNode
from parser.tokens import NumberToken, IdentNameToken


class Test(TestCase):
    def test_node_from_token(self):
        null_region = StrRegion(0, 0)
        self.assertIsInstance(node_from_token(NumberToken()), NumberNode)
        self.assertEqual(NumberNode(null_region), node_from_token(NumberToken()))
        self.assertIsInstance(node_from_token(IdentNameToken()), IdentNode)
        self.assertEqual(IdentNode(null_region), node_from_token(IdentNameToken()))
        self.assertEqual(IdentNode(StrRegion(5, 7)),
                         node_from_token(IdentNameToken(StrRegion(5, 7))))

    def test_node_add(self):
        nd = Node(StrRegion(0, 6))
        lf1 = Leaf(StrRegion(0, 2))
        lf2 = Leaf(StrRegion(3, 5))
        nd.add(lf1, lf2)
        self.assertEqual(nd.children, [lf1, lf2])

    def test_node_init_with_children(self):
        lf1 = Leaf(StrRegion(0, 2))
        lf2 = Leaf(StrRegion(3, 5))
        nd = Node(StrRegion(0, 6), [lf1, lf2])
        self.assertEqual(nd.children, [lf1, lf2])
