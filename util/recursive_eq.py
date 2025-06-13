"""Safe recursive equality comparison."""
import functools


def recursive_eq(fn):
    """Must be used as decorator, like reprlib.recursive_repr.
    Works by hypothesising that 2 ids are equal. Then, it tries to compare
    them. If it encounters one of them again, it checks that the corresponding
    value is the hypothesised value. If so, they're equal. If not, they're
    unequal."""
    hypotheses: dict[int, int] = {}  # int <-> int (should be undirected)

    @functools.wraps(fn)
    def eq(a, b):
        if (bid_exp := hypotheses.get(id(a))) is not None:
            return bid_exp == id(b)
        if (aid_exp := hypotheses.get(id(b))) is not None:
            return aid_exp == id(a)
        hypotheses[id(a)] = id(b)
        hypotheses[id(b)] = id(a)
        try:
            return fn(a, b)  # Will call this function again
        finally:
            del hypotheses[id(a)]
            del hypotheses[id(b)]
    return eq
