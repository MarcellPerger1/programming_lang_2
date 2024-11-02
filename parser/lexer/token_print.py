from __future__ import annotations

import sys
from io import StringIO
from typing import IO

from .tokens import Token


def print_tokens(src: str, tokens: list[Token], stream: IO[str] = None, do_ws=False):
    if stream is None:
        stream = sys.stdout
    table = []
    for tok in tokens:
        if tok.is_whitespace:
            if do_ws:
                table.append(['(WS) ' + repr(tok.region.resolve(src)), tok.name])
        else:
            table.append([str(tok.region.resolve(src)), tok.name])
    max0 = max(len(r[0]) for r in table)
    max1 = max(len(r[1]) for r in table)
    for s0, s1 in table:
        print(f'{s0:>{max0}} | {s1:>{max1}}', file=stream)


def format_tokens(src: str, tokens: list[Token], do_ws=False):
    out = StringIO()
    print_tokens(src, tokens, out, do_ws)
    return out.getvalue()
