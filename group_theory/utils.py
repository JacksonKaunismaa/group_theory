from collections import deque
import itertools
from typing import TYPE_CHECKING
from tqdm import tqdm

if TYPE_CHECKING:
    from group_theory import groups


def sliding_window(iterable, n):  # standard itertools recipe
    it = iter(iterable)
    window = deque(
        itertools.islice(it, n), maxlen=n
    )  # maxlen => the first element of the window
    if len(window) == n:  # will be removed when we do the append operation
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


def factorize(n):
    # from https://codereview.stackexchange.com/questions/121862/fast-number-factorization-in-python
    # some common primes, should mean we can handle reasonable cases very slightly faster
    primes = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]
    for fact in itertools.chain(primes, itertools.count(primes[-1], 2)):
        if n == 1:
            break
        while n % fact == 0:
            n //= fact
            yield fact
        if fact**2 > n:  # this means n must be prime
            yield n
            break


def get_interesting_sizes(n):
    # return a list of exponents that would generate a different subgroup from a single element of order n
    # this only really works for single elements
    # factors = list(Counter(factorize(n)).items())  # so that order is fixed
    # yield 0  # give the option to not use this particular element
    # for exponents in  itertools.product(*(range(sz) for _,sz in factors)):
    #     elem = 1
    #     for (fact,_), amt in zip(factors, exponents):
    #         elem *= fact**amt
    #     yield elem

    return range(n)  # this is the actual safest


def find_subgroups(group: "groups.Group"):  # only works for finite groups
    generators = group.generators()
    generators = list(generators.items())  # so that order is fixed
    subgroups = {}
    # iterate over all possible exponents for all possible generators
    generator_iter = itertools.product(
        *(get_interesting_sizes(order) for _, order in generators)
    )
    # generate each subgroup using the same number of generators as there are in the full group
    # if the same generator is repeated, that is equivalent to only using 1 generator
    # if generators are specified in a different order, they are equivalent to each other as well
    # thus, it makes the most sense to consider subgroup generators as a set() object, but we want them to be
    # hashable as well, so make it a frozenset
    for generator_terms in tqdm(
        itertools.product(generator_iter, repeat=len(generators))
    ):
        # inner comprehension is multiplying together generators of the full group to get one of the generators for the subgroup
        subgroup_generators = frozenset(
            group.parse(
                " ".join(
                    f"{elem}{amt}" for (elem, _), amt in zip(generators, generator_term)
                )
            )
            for generator_term in generator_terms
        )
        # outer comprehension is repeating that process for all the subgroup generators

        if subgroup_generators not in subgroups:
            new_subgroup = group.generate(*subgroup_generators)
            # print(amts, "<" + ", ".join(str(x) for x in subgroup_generators) + ">", "=", new_subgroup)
            if (
                new_subgroup and new_subgroup not in subgroups.values()
            ):  # only add unique subgroups
                subgroups[subgroup_generators] = new_subgroup
    return subgroups
