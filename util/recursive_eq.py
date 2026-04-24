"""Safe recursive equality comparison."""
import functools


# TODO: this probably has a bug when `a is b` (thanks gpt-5.5 - it's literally
#  the only model that spots it (I made 12 models them generate tests to see
#  how they' doing nowadays via LMArena, with somewhat adequate results),
#  no other one even thinks to test this case. If gpt has the intuition to
#  think 'what is their identical', then that is genuinely terrifying for
#  my job prospects.
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
