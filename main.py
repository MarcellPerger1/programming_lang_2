import cProfile
import contextlib
import time

from parser.astgen.ast_node import AstProgramNode
from parser.astgen.astgen import AstGen
from parser.common.tree_print import tformat
from parser.cst.nodes import ProgramNode
from parser.cst.treegen import TreeGen
from parser.lexer import Tokenizer, format_tokens
from util import readfile

PROFILER = True


class _Timer:
    _start = _end = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.perf_counter()

    def get(self):
        return self._end - self._start


class PerfOnce:
    _tokenizer: Tokenizer
    _cstgen: TreeGen
    _cst: ProgramNode
    _ast: AstProgramNode

    def __init__(self, src: str, idx: int = -1, do_ast=True):
        self.src = src
        self.idx = idx
        self.should_do_ast = do_ast
        self.lines: list[tuple[float, str]] = []  # First item used as key

    @classmethod
    def _fmt_time_taken(cls, name: str, delta_sec: float):
        return f'{name:<17} done in {delta_sec * 1000:.2f}ms'

    @classmethod
    def _maybe_profiler(cls):
        if PROFILER:
            return cProfile.Profile()
        return contextlib.nullcontext(None)

    def run(self):
        with self._maybe_profiler() as p:
            self.do_tokenize()
            self.do_token_fmt()
            self.do_cst()
            self.do_cst_fmt()
            if self.should_do_ast:
                self.do_ast()
                self.do_ast_fmt()
        if p:
            p.dump_stats(f'perf_dump_{self.idx}.prof')
        print(f'Perf for idx={self.idx} ({PROFILER=}):')
        for _k, s in sorted(self.lines):
            print(f'  {s}')

    def _add_line(self, sort_key: float, name: str, delta_sec: float):
        self.lines.append((sort_key, self._fmt_time_taken(name, delta_sec)))

    def do_tokenize(self):
        with _Timer() as t:
            self._tokenizer = Tokenizer(self.src).tokenize()
        self._add_line(0.0, 'Tokens', t.get())

    def do_token_fmt(self):
        with _Timer() as t:
            _s = format_tokens(self.src, self._tokenizer.tokens, True)
        self._add_line(0.5, 'Tokens_fmt', t.get())

    def do_cst(self):
        with _Timer() as t:
            self._cstgen = TreeGen(self._tokenizer)
            self._cst = self._cstgen.parse()
        self._add_line(1.0, 'CST', t.get())

    def do_cst_fmt(self):
        with _Timer() as t:
            _s = tformat(self._cst)
        self._add_line(1.5, 'CST', t.get())

    def do_ast(self):
        with _Timer() as t:
            self._ast = AstGen(self._cstgen).parse()
        self._add_line(2.0, 'AST', t.get())

    def do_ast_fmt(self):
        with _Timer() as t:
            _s = tformat(self._ast)
        self._add_line(2.5, 'AST', t.get())


def run(src: str, idx: int = -1, do_ast=True):
    return PerfOnce(src, idx, do_ast).run()


def main():
    run(readfile('main_example_0.st'), 0, do_ast=False)
    run(readfile('main_example_1.st'), 1)


if __name__ == '__main__':
    main()
