import random
import string
import time

from parser.astgen.ast_node import AstNode
from parser.astgen.astgen import AstGen
from parser.astgen.filtered_walker import FilteredWalker
from parser.lexer.tokenizer import Tokenizer
from parser.cst.cstgen import CstGen
from parser.common.error import BaseParseError
from parser.typecheck.typecheck import Typechecker
from parser.typecheck.name_resolver import NameResolver
from parser.typecheck.types import TypeMetadata

from pythonfuzz.fuzzer import Fuzzer
import pythonfuzz.fuzzer as fuzzer_ns  # For patching pythonfuzz

VERSION = 2


class UsePerfCounterInsteadOfTime:
    """Hack to avoid overwriting everyone's time module so we only
    overwrite `pythonfuzz`'s time module and don't modify the module itself.
    This hack is necessary because time.time() is rather inaccurate so
    it is possible that between 2 iterations, the difference in time.time() is 0
    which results in DivisionByZeroError (when calculating iterations/sec).
    Therefore, we replace with the more accurate time.perf_counter(),
    just for `pythonfuzz`"""
    def __getattr__(self, item):
        if item == 'time':  # time.time
            item = 'perf_counter'
        return getattr(time, item)


fuzzer_ns.time = UsePerfCounterInsteadOfTime()


def assert_all_metadata(n: AstNode):
    def on_exit_node(nd: AstNode):
        assert isinstance(nd.meta, TypeMetadata)

    # Report error with deepest one so on_exit
    FilteredWalker().register_exit(AstNode, on_exit_node).walk(n)
    return n


def fuzz_advanced(buf: bytes):
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
                 + i.to_bytes(length=i.bit_count() // 8 + 1))
        parts.append(rng.choice(sample_from))
    s = ''.join(parts)
    target(s)


def fuzz(buf: bytes):
    if VERSION == 2:
        if not buf:
            return
        flag, *buf = buf
        if flag.bit_count():  # Parity rather than MSB check for less bias?
            return fuzz_advanced(bytes(buf))
        buf = bytes(map(0x7F.__and__, buf))  # wdym you don't understand
    try:
        s = buf.decode("ascii")
    except UnicodeDecodeError:
        return  # Shouldn't happen though will catch anyway
    target(s)


def target(s: str):
    try:
        n = Typechecker(NameResolver(AstGen(CstGen(Tokenizer(s))))).run()
    except BaseParseError:
        return
    assert_all_metadata(n)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser("fuzz.py", description="Runs a fuzzer for n iterations")
    # Use type=float as gh mobile cannot specify integers as workflow args
    ap.add_argument('-n', '--iterations', default=-1,
                    type=float, help="Number of iterations to run pythonfuzz for")
    ap.add_argument('-i', '--infinite',
                    action='store_const', const=-1, dest='iterations')
    args = ap.parse_args()

    fuzzer = Fuzzer(fuzz, dirs=['./pythonfuzz_corpus'], timeout=30, runs=int(args.iterations))
    fuzzer.start()
