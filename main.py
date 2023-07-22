import time

from parser.cst.treegen import TreeGen
from parser.tokenizer import Tokenizer, print_tokens
from parser.cst.tree_print import tprint


def make_tree(src: str):
    return TreeGen(Tokenizer(src)).parse()


def read_file(path: str):
    with open(path) as f:
        return f.read()


PROFILER = True


def run(src: str, idx: int = -1):
    import cProfile
    tn0 = time.perf_counter()
    tn = Tokenizer(src).tokenize()
    tn1 = time.perf_counter()
    print('Tokens:')
    print_tokens(tn.src, tn.tokens)
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
    tprint(node)
    print(f'Tokens done in {(tn1 - tn0) * 1000:.2f}ms')
    print(f'CST    done in {(tp1 - tp0) * 1000:.2f}ms')


def main():
    run(read_file('main_example_0.txt'), 0)
    run(read_file('main_example_1.txt'), 1)


if __name__ == '__main__':
    main()
