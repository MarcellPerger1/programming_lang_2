import enum
from unittest import TestCase

from parser.common import StrRegion
from parser.common.tree_print import tformat
from parser.cst import nodes as cst_nodes
from parser.astgen import ast_nodes
from parser.typecheck.typecheck import ValType, FunctionType, ListType, VoidType


class _DummyEnum(enum.Enum):
    A = 1
    B = 2


class TestTreePrinter(TestCase):
    def assertFormatsTo(self, obj: object, expect: str):
        actual = tformat(obj, indent=2, verbose=True, append_lf=False)
        self.assertEqual(expect, actual)

    def test_elemental(self):
        self.assertFormatsTo(123, '123')
        self.assertFormatsTo(_DummyEnum.A, '_DummyEnum.A')
        self.assertFormatsTo(StrRegion(3, 5), 'StrRegion(3, 5)')

    def test_empty_collections(self):
        self.assertFormatsTo([], '[]')
        self.assertFormatsTo((), '()')

    def test_flat_element_collections(self):
        self.assertFormatsTo([2], '[\n  2\n]')
        self.assertFormatsTo((2,), '(\n  2,\n)')
        self.assertFormatsTo([2, "aa", StrRegion(3, 5)],
                             "[\n  2,\n  'aa',\n  StrRegion(3, 5)\n]")

    def test_cst_nodes(self):
        ae = cst_nodes.AddEqNode(StrRegion(0, 10), None, [
            cst_nodes.IdentNode(StrRegion(0, 2)),
            cst_nodes.SubNode(StrRegion(4, 10), None, [
                cst_nodes.IdentNode(StrRegion(4, 5)),
                cst_nodes.UMinusNode(StrRegion(6, 10), None, [
                    cst_nodes.NumberNode(StrRegion(7, 10))
                ])
            ])
        ])
        self.assertFormatsTo(
            ae, dedent('''\
                AddEqNode(StrRegion(0, 10), [
                  IdentNode(StrRegion(0, 2)),
                  SubNode(StrRegion(4, 10), [
                    IdentNode(StrRegion(4, 5)),
                    UMinusNode(StrRegion(6, 10), [
                      NumberNode(StrRegion(7, 10))
                    ])
                  ])
                ])'''))

    def test_ast_nodes(self):
        self.assertFormatsTo(ast_nodes.AstIdent(StrRegion(0, 2), 'ab'),
                             "AstIdent(StrRegion(0, 2), 'ab')")
        n = ast_nodes.AstAssign(
            StrRegion(0, 10),
            ast_nodes.AstIdent(StrRegion(0, 2), 'ab'),
            ast_nodes.AstBinOp(
                StrRegion(3, 10),
                '+',
                ast_nodes.AstIdent(StrRegion(3, 4), 'c'),
                ast_nodes.AstUnaryOp(
                    StrRegion(5, 10),
                    '-',
                    ast_nodes.AstNumber(StrRegion(6, 10), 1000)
                )
            )
        )
        self.assertFormatsTo(n, dedent('''\
            AstAssign(StrRegion(0, 10),
              AstIdent(StrRegion(0, 2), 'ab'),
              AstBinOp(StrRegion(3, 10),
                '+',
                AstIdent(StrRegion(3, 4), 'c'),
                AstUnaryOp(StrRegion(5, 10),
                  '-',
                  AstNumber(StrRegion(6, 10), 1000)
                )
              )
            )'''))

    def test_typed_ast_nodes(self):
        self.assertFormatsTo(ast_nodes.AstIdent(StrRegion(0, 2), 'ab', meta=ValType()),
                             "AstIdent(StrRegion(0, 2), 'ab', meta=ValType())")
        n = ast_nodes.AstAssign(
            StrRegion(0, 10),
            ast_nodes.AstIdent(StrRegion(0, 2), 'ab'),
            ast_nodes.AstBinOp(
                StrRegion(3, 10),
                '+',
                ast_nodes.AstIdent(StrRegion(3, 4), 'c'),
                ast_nodes.AstUnaryOp(
                    StrRegion(5, 10),
                    '-',
                    ast_nodes.AstNumber(StrRegion(6, 10), 1000),
                    meta=FunctionType([ListType(), ValType()], VoidType())
                )
            ),
            meta=ValType()
        )
        self.assertFormatsTo(n, dedent('''\
            AstAssign(StrRegion(0, 10),
              AstIdent(StrRegion(0, 2), 'ab'),
              AstBinOp(StrRegion(3, 10),
                '+',
                AstIdent(StrRegion(3, 4), 'c'),
                AstUnaryOp(StrRegion(5, 10),
                  '-',
                  AstNumber(StrRegion(6, 10), 1000),
                  meta=FunctionType(arg_types=[ListType(), ValType()], ret_type=VoidType())
                )
              ),
              meta=ValType()
            )'''))


def _indent_size(line: str):
    return len(line) - len(line.lstrip())


def dedent(s: str):
    s = s.removeprefix('\n')  # if '''<newline> and forgotten \ after the '''
    content_lines = filter(None, map(str.rstrip, s.splitlines()))
    amount = min(map(_indent_size, content_lines))
    result_lines = []
    for line in s.split('\n'):  # not splitlines to preserve possible newline at EOF
        num_ws = len(line) - len(line.lstrip())
        result_lines.append(line[min(amount, num_ws):])  # don't strip more WS than there is
    return '\n'.join(result_lines)
