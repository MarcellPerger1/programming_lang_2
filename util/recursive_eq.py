"""Safe recursive equality comparison."""
import functools


def recursive_eq(fn):
    """Allows safe recursive equality comparisons also cnosidering reference
    identity (like if I modify a will b change?). Must be used as decorator,
    like reprlib.recursive_repr. Note: may break with multi-threading or
    super-weird reentrancy (comparing unrelated objects still of the same
    type in the __eq__ where it has no relation to the current comparison).

    Works by hypothesising that 2 ids are equal. Then, it tries to compare
    them. If it encounters one of them again, it checks that the corresponding
    value is the hypothesised value. If so, they're equal. If not, they're
    unequal."""
    hypotheses: dict[int, int] = {}  # int <-> int (should be undirected)

    @functools.wraps(fn)
    def eq(a, b):
        if a is b:
            return True  # Prevents nasty stuff like deleting same key twice
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
