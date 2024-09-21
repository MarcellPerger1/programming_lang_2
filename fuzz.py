import time

from parser.lexer.tokenizer import Tokenizer
from parser.cst.treegen import TreeGen
from parser.error import BaseParseError

from pythonfuzz.fuzzer import Fuzzer
import pythonfuzz.fuzzer as fuzzer_ns  # For patching pythonfuzz


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


def fuzz(buf):
    try:
        string = buf.decode("ascii")
        try:
            TreeGen(Tokenizer(string)).parse()
        except BaseParseError:
            pass
    except UnicodeDecodeError:
        pass


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser("fuzz.py", description="Runs a fuzzer for n iterations")
    ap.add_argument('-n', '--iterations', default=-1,
                    type=int, help="Number of iterations to run pythonfuzz for")
    ap.add_argument('-i', '--infinite',
                    action='store_const', const=-1, dest='iterations')
    args = ap.parse_args()

    fuzzer = Fuzzer(fuzz, dirs=['./pythonfuzz_corpus'], timeout=30, runs=args.iterations)
    fuzzer.start()
