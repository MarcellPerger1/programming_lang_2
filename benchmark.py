import cProfile
import contextlib
import time

from parser.astgen.ast_node import AstProgramNode
from parser.astgen.astgen import AstGen
from parser.common.tree_print import tformat
from parser.cst.nodes import ProgramNode
from parser.cst.cstgen import CstGen
from parser.lexer import Tokenizer, format_tokens
from parser.typecheck.typecheck import NameResolver, Scope
from util import readfile
from util.pformat import pformat

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


class BenchOnce:
    _tokenizer: Tokenizer
    _cstgen: CstGen
    _cst: ProgramNode
    _astgen: AstGen
    _ast: AstProgramNode
    _nr: NameResolver
    _top_scope: Scope

    def __init__(self, src: str, idx: int = -1, do_ast=True,
                 do_name_resolve=True):
        self.src = src
        self.idx = idx
        self.should_do_ast = do_ast
        self.should_name_resolve = do_name_resolve
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
                if self.should_name_resolve:
                    self.do_name_resolve()
                    self.do_name_resolve_fmt()
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
            self._cstgen = CstGen(self._tokenizer)
            self._cst = self._cstgen.parse()
        self._add_line(1.0, 'CST', t.get())

    def do_cst_fmt(self):
        with _Timer() as t:
            _s = tformat(self._cst)
        self._add_line(1.5, 'CST_fmt', t.get())

    def do_ast(self):
        with _Timer() as t:
            self._astgen = AstGen(self._cstgen)
            self._ast = self._astgen.parse()
        self._add_line(2.0, 'AST', t.get())

    def do_ast_fmt(self):
        with _Timer() as t:
            _s = tformat(self._ast)
        self._add_line(2.5, 'AST_fmt', t.get())

    def do_name_resolve(self):
        with _Timer() as t:
            self._nr = NameResolver(self._astgen)
            self._top_scope = self._nr.run()
        self._add_line(3.0, 'NameRes', t.get())

    def do_name_resolve_fmt(self):
        with _Timer() as t:
            _s = pformat(self._top_scope)
        self._add_line(3.5, 'NameRes_fmt', t.get())


def benchmark(src: str, idx: int = -1, do_ast=True, do_name_resolve=True):
    return BenchOnce(src, idx, do_ast, do_name_resolve).run()


def bench_full(n=200):
    times = []
    # noinspection PyProtectedMember
    with BenchOnce._maybe_profiler() as p:
        for _ in range(n):
            t0 = time.perf_counter()
            _sc = NameResolver(AstGen(CstGen(Tokenizer(
                readfile('main_example_2.st'))))).run()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    if p:
        p.dump_stats('./long_perf.prof')
    print(f'Bench main_example_2.st, {n} iterations, ({PROFILER=}):')
    print(f'  Min: {min(times)*1000:.2f}ms')
    print(f'  Avg: {sum(times)/n*1000:.2f}ms')


def main():
    benchmark(readfile('main_example_0.st'), 0, do_ast=False)
    benchmark(readfile('main_example_1.st'), 1, do_name_resolve=False)
    benchmark(readfile('main_example_2.st'), 2)
    bench_full(200)


if __name__ == '__main__':
    main()
