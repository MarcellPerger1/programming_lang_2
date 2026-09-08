import random
import string

from compiler.astgen.ast_node import AstNode
from compiler.astgen.astgen import AstGen
from compiler.astgen.filtered_walker import FilteredWalker
from compiler.common.error import BaseParseError
from compiler.cst.cstgen import CstGen
from compiler.lexer.tokenizer import Tokenizer
from compiler.typecheck.name_resolver import NameResolver
from compiler.typecheck.typecheck import Typechecker
from compiler.typecheck.types import TypeMetadata

__all__ = ['fuzz_target', 'fuzz_target_string']


def fuzz_target(buf: bytes, version=2):
    if version == 2:
        if not buf:
            return
        flag, *buf = buf
        if flag.bit_count() % 2 == 0:  # Parity rather than MSB check for less bias?
            return _fuzz_advanced(bytes(buf))
        buf = bytes(map(0x7F.__and__, buf))  # wdym you don't understand
    try:
        s = buf.decode("ascii")
    except UnicodeDecodeError:
        return  # Shouldn't happen though will catch anyway
    fuzz_target_string(s)


def _fuzz_advanced(buf: bytes):
    # TODO There are massive biases due to Pythonfuzz's compulsion to give us
    #  specific byte values a lot more (stuff like FF, 00, 7F, 80)
    important = [
        'let', 'for', 'while', 'if', 'else', 'global', *'+-*/%=<>!&|', 'def',
        'repeat', ';', 'val', 'number', 'string', 'bool', *'[](){}', "'", '"',
        '\\', '.', *(' ' * 30)  # same weight of spaces as other stuff approx
    ]
    other = [*string.printable]
    sample_from = important * 5 + other
    parts = []
    rng = random.Random()
    for i, byte in enumerate(buf):
        rng.seed(buf + b'\0' + bytes(byte) + b'\0'
                 + i.to_bytes(length=i.bit_count() // 8 + 1, byteorder='big'))
        parts.append(rng.choice(sample_from))
    s = ''.join(parts)
    fuzz_target_string(s)


def fuzz_target_string(s: str):
    try:
        n = Typechecker(NameResolver(AstGen(CstGen(Tokenizer(s))))).run()
    except BaseParseError:
        return
    assert_all_metadata(n)


def assert_all_metadata(n: AstNode):
    def on_exit_node(nd: AstNode):
        assert isinstance(nd.meta, TypeMetadata)

    # Report error with deepest one so on_exit
    FilteredWalker().register_exit(AstNode, on_exit_node).walk(n)
    return n
