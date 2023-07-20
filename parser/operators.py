from __future__ import annotations


# no @ or // for now
# no bitwise operators for now
# .. for concat
ASSIGNABLE_OPS = (*'+-*/%', '**', '&&', '||', '..')
ASSIGN_OPS = tuple(o + '=' for o in ASSIGNABLE_OPS) + ('=',)
COMPARISONS = ('==', '!=', '<=', '>=', '<', '>')
UNARY_OPS = (*'!+-',)
BINARY_OPS = (*'+-*/%', '**', '&&', '||')
# all the ops that are actual single-token operators
# and require no special syntax considerations
OPS_SET = frozenset(UNARY_OPS + BINARY_OPS + COMPARISONS + ASSIGN_OPS)
ALL_OPS = tuple(OPS_SET)
SORTED_OPS = tuple(sorted(ALL_OPS, key=lambda o: (len(o), o), reverse=True))
MAX_OP_LEN = len(SORTED_OPS[0])
OP_FIRST_CHARS: frozenset[str] = frozenset({o[0] for o in SORTED_OPS})
