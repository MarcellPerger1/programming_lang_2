import cProfile
import time

from parser.astgen.astgen import AstGen
from util import readfile
from parser.cst.treegen import TreeGen
from parser.lexer import Tokenizer, print_tokens
from parser.common.tree_print import tprint


def make_tree(src: str):
    return TreeGen(Tokenizer(src)).parse()


PROFILER = True


def run(src: str, idx: int = -1, do_ast=True):
    node = ast_node = None
    ta1 = tp1 = ta0 = 0.0  # will be overwritten

    def doit_trees():
        nonlocal node, tp1, ta0, ast_node, ta1
        treegen = TreeGen(tn)
        node = treegen.parse()
        tp1 = time.perf_counter()
        if do_ast:
            ta0 = time.perf_counter()
            ast_node = AstGen(treegen).parse()
            ta1 = time.perf_counter()

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
            doit_trees()
        p.dump_stats(f'perf_dump_{idx}.prof')
    else:
        doit_trees()
    print('CST:')
    tpr_cst0 = time.perf_counter()
    tprint(node)
    tpr_cst1 = time.perf_counter()
    tpr_ast0 = tpr_ast1 = time.perf_counter()
    if do_ast:
        tprint(ast_node)
        tpr_ast1 = time.perf_counter()
    print(rf'Tokens            done in {(tn1 - tn0) * 1000:.2f}ms')
    print(rf'Tokens_print      done in {(tpr_tk1 - tpr_tk0) * 1000:.2f}ms')
    print(rf'CST               done in {(tp1 - tp0) * 1000:.2f}ms')
    print(rf'CST_print         done in {(tpr_cst1 - tpr_cst0) * 1000:.2f}ms')
    if do_ast:
        print(rf'AST               done in {(ta1 - ta0) * 1000:.2f}ms')
        print(rf'AST_print         done in {(tpr_ast1 - tpr_ast0) * 1000:.2f}ms')


def main():
    run(readfile('main_example_0.st'), 0, do_ast=False)
    run(readfile('main_example_1.st'), 1)


if __name__ == '__main__':
    main()
