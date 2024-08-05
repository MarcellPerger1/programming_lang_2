import multiprocessing

orig_ssm = multiprocessing._set_start_method = multiprocessing.set_start_method


def new_ssm(m: str):
    if m != 'fork':
        orig_ssm(m)


multiprocessing.set_start_method = new_ssm  # Monkey-patch


# TODO: make this work on CI?
# To get ths to work (Win11): Change
# execs_per_second = int(self._executions_in_sample / (endTime - self._last_sample_time))
# to
# execs_per_second = int(self._executions_in_sample / (endTime - self._last_sample_time or 0.000001))

from pythonfuzz.main import PythonFuzz

from parser.lexer.tokenizer import Tokenizer
from parser.cst.treegen import TreeGen
from parser.error import BaseParseError


def fuzz(buf):
    try:
        string = buf.decode("ascii")
        try:
            TreeGen(Tokenizer(string)).parse()
        except BaseParseError:
            pass
    except UnicodeDecodeError:
        pass


# pickling errors... (old func would be inaccessible using decorator)
fuzz2 = PythonFuzz(fuzz)


if __name__ == '__main__':
    fuzz2()
