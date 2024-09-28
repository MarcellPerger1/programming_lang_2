import cProfile
import time

from parser.util import readfile
from parser.cst.treegen import TreeGen
from parser.lexer import Tokenizer, print_tokens
from parser.cst.tree_print import tprint


def make_tree(src: str):
    return TreeGen(Tokenizer(src)).parse()


PROFILER = True


def run(src: str, idx: int = -1):
    tn0 = time.perf_counter()
    tn = Tokenizer(src).tokenize()
    tn1 = time.perf_counter()
    print('Tokens:')
    tpr_tk0 = time.perf_counter()
    print_tokens(tn.src, tn.tokens)
    tpr_tk1 = time.perf_counter()
    tp0 = time.perf_counter()
    if PROFILER:
        with cProfile.Profile() as p:
            node = TreeGen(tn).parse()
        tp1 = time.perf_counter()
        p.dump_stats(f'perf_dump_{idx}.prof')
    else:
        node = TreeGen(tn).parse()
        tp1 = time.perf_counter()
    print('CST:')
    tpr_cst0 = time.perf_counter()
    tprint(node)
    tpr_cst1 = time.perf_counter()
    print(rf'Tokens            done in {(tn1 - tn0) * 1000:.2f}ms')
    print(rf'Tokens_print      done in {(tpr_tk1 - tpr_tk0) * 1000:.2f}ms')
    print(rf'CST               done in {(tp1 - tp0) * 1000:.2f}ms')
    print(rf'CST_print         done in {(tpr_cst1 - tpr_cst0) * 1000:.2f}ms')


def main():
    run(readfile('main_example_0.st'), 0)
    run(readfile('main_example_1.st'), 1)


if __name__ == '__main__':
    main()
