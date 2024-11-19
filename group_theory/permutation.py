import itertools
from typing import Any, Sequence, Union, List


from .group_element import GroupElement
from .groups import Group

# the types that we can try to parse as Permutation in permissive multiplication
PermutationFormat = Union[Sequence[int], List[List[int]], str]
PermissivePerm = Union[PermutationFormat, "Permutation"]


class PermutationGroup(Group["Permutation"]):
    """
    A class representing a permutation group, which is a group consisting of permutations.

    Attributes:
        n (int): The degree of the permutation group.
        name (str): The name of the permutation group.
        verbose (bool): If True, enables verbose output.
    """

    def __init__(
        self,
        *elems: "Permutation",
        n: int,
        name: str,
        generate: bool = False,
        verbose: bool = False,
    ):
        super().__init__(*elems, name=name, verbose=verbose)
        self.n = n
        if generate:
            self._generate_all()

    def _generate_all(self):
        new_elems = [
            Permutation(pt, self, True).simplify()
            for pt in itertools.permutations(list(range(self.n)))
        ]

        if "symmetric" not in self.name:
            new_elems = [x for x in new_elems if x.parity == 0]

        self.update(new_elems)

    def _same_group_type(self, other: Group):
        types_match = super()._same_group_type(other)
        if types_match:
            return self.n == other.n
        return False

    def identity_expr(self):
        return Permutation([], self, True)

    def parse(self, equation: PermutationFormat, initial=False) -> "Permutation":
        return Permutation(equation, self)

    def generators(self):
        # list of adjacent transpositions
        return [Permutation([[a, a + 1]], self, True) for a in range(0, self.n - 1)]

    def subgroup(self, *elems: PermissivePerm) -> "PermutationGroup":
        perm_elems = self.evaluates(*elems)
        group = PermutationGroup(
            *perm_elems, n=self.n, name=self.name, verbose=self.verbose, generate=False
        )
        group.copy_subgroup_attrs_to(self)
        return group


class Permutation(GroupElement):
    """
    Represents a permutation in a permutation group.

    Attributes:
        group (PermutationGroup): The group to which this permutation belongs.
        cycle (List[List[int]]): The cycle notation of the permutation.
    """

    def __init__(
        self,
        notation: PermutationFormat,
        group: "PermutationGroup",
        is_valid: bool = False,
    ):
        # notation must be one of cycle notation or result notation
        # cycle notation is identified as a list[list[int]], result notation is list[int]
        self.group = group

        if isinstance(notation, str):
            self.cycle = self._parse(notation)
        elif notation and isinstance(notation[0], int):
            self.parse_result_notation(notation)  # type: ignore
        elif notation and isinstance(notation[0], list):
            self.cycle: List[List[int]] = notation
            if not is_valid:
                self.validate()
        elif not notation:
            self.cycle = []
        else:
            raise TypeError(f"Invalid notation {type(notation)}")

    def validate(self):
        for term in self.cycle:
            for i in term:
                if i >= self.group.n:
                    raise ValueError(
                        f"Invalid permutation: {i} is too large in {term}."
                    )

    def parse_result_notation(self, notation: Sequence[int]):
        shifted_notation = notation
        remain = set(range(self.group.n))
        curr_term = []
        self.cycle = []
        elem = self.start_new_term(remain, curr_term)  # [2, 5, 3, 1, 4]
        while remain:  # [1,4,2,0,3])
            # print(elem, "goes to", end=" ")
            elem = shifted_notation.index(elem)
            # print(elem)
            if elem == curr_term[0]:
                # print(f"cycle finished {curr_term=}")
                self.cycle.append(curr_term.copy())
                curr_term = []
                elem = self.start_new_term(remain, curr_term)
            else:
                curr_term.append(elem)
                remain.remove(elem)
        self.cycle.append(curr_term.copy())

    def start_new_term(self, remain: set[int], curr_term: list[int]) -> int:
        elem = min(remain)
        curr_term.append(elem)
        remain.remove(elem)
        return elem

    def _parse(self, equation: str, initial=False) -> List[List[int]]:
        cycles = equation.split("(")
        cycles = [x.strip("() ") for x in cycles if x]
        cycles = [[int(c) - 1 for c in x.split(" ") if c] for x in cycles]
        return cycles

    @property
    def is_identity(self):
        return not self.cycle

    @property
    def cycle_type(self):
        cycle_lens = [len(x) for x in self.cycle]
        num_cycles = []
        for i in range(self.group.n):  # the 1's count will likely be wrong
            num_cycles.append(cycle_lens.count(i + 1))
        return num_cycles

    @property
    def parity(self):
        return sum(i * amt for i, amt in enumerate(self.cycle_type)) % 2

    def __repr__(self):
        if not self.cycle:
            return "e"
        else:
            return "".join(
                [f'({" ".join(map(lambda y: str(y+1), x))})' for x in self.cycle]
            )

    def result_notation(self):
        """Returns the result notation of the permutation."""
        result = list(range(self.group.n))
        for cycle in self.cycle:
            for i in range(len(cycle)):
                result[cycle[(i + 1) % len(cycle)]] = cycle[i]
        return result

    def simplify(self):
        # print(self, self.group.name)
        # print("cycle begins as", self.cycle)
        remain = set(range(self.group.n))
        new_cycle = []
        curr_term = []
        elem = self.start_new_term(remain, curr_term)
        while remain:  # (0 2 3)(1 2)(3)(3 2)
            for term in self.cycle:
                # print("Term iis ", term)
                term_size = len(term)
                try:
                    loc = term.index(elem)
                    # print(elem, "goes to", end=" ")
                    elem = term[(loc + 1) % term_size]
                    # print(elem, f"({term=})")
                except ValueError:
                    continue
            if elem == curr_term[0]:
                # print(f"finished cycle {curr_term=}")
                new_cycle.append(curr_term.copy())
                curr_term = []
                elem = self.start_new_term(remain, curr_term)
            else:
                curr_term.append(elem)
                remain.remove(elem)
        # print("simplified cycle", new_cycle)
        new_cycle.append(curr_term.copy())
        filtered = self._filter_identity(new_cycle)
        if self.group.quotient_map:
            return self.group.quotient_map[
                filtered
            ]  # don't bother checking existence, since that should throw an error anyway
        return filtered

    def _filter_identity(self, cycle=None):
        if cycle is None:
            cycle = self.cycle
        return Permutation([x for x in cycle if len(x) > 1], self.group, True)

    def inv(self):
        return Permutation(
            list(reversed([list(reversed(x)) for x in self.cycle])), self.group, True
        ).simplify()

    def permissive(self, other: PermissivePerm) -> "Permutation":
        """
        Try to promote expressions to Permutation for permissive multiplication.

        1. If not an Permutation, try converting Permutation
        2. If this fails, __init__ will throw a TypeError
        3. Otherwise, return the Permutation
        """
        if not isinstance(other, Permutation):
            other = Permutation(other, self.group)  # type: ignore
        return other

    def __mul__(self, other: PermissivePerm) -> "Permutation":
        # if not isinstance(other, Permutation):
        #     other = Permutation(other, self.group)
        # cycle = self.cycle + other.cycle
        # return Permutation(cycle, self.group, True).simplify()
        try:
            other = self.permissive(other)
            cycle = self.cycle + other.cycle
            return Permutation(cycle, self.group, True).simplify()
        except TypeError:
            return NotImplemented

    def __rmul__(self, other: PermissivePerm):
        try:
            other = self.permissive(other)
            cycle = other.cycle + self.cycle
            return Permutation(cycle, self.group, True).simplify()
        except TypeError:
            return NotImplemented

    def __hash__(self):
        return hash(str(self))

    def __truediv__(self, other: PermissivePerm) -> "Permutation":
        try:
            other = self.permissive(other)
            return self * other.inv()
        except TypeError:
            return NotImplemented

    def __rtruediv__(self, other: PermissivePerm) -> "Permutation":
        try:
            other = self.permissive(other)
            return other * self.inv()
        except TypeError:
            return NotImplemented

    def __eq__(self, other: Any) -> bool:
        try:
            other = self.permissive(other)
            return str(self) == str(other)
        except TypeError:
            return NotImplemented

    def simpler_heuristic(self, other: "Permutation") -> bool:
        if self.is_identity or other.is_identity:
            return self.is_identity

        sum_cycle1, sum_cycle2 = sum(self.cycle_type), sum(other.cycle_type)
        if sum_cycle1 < sum_cycle2:  # fewer terms are preferred
            return True
        elif sum_cycle1 == sum_cycle2:  # shorter terms are preferred
            total = sum(
                i * (c1 - c2)
                for i, (c1, c2) in enumerate(zip(self.cycle_type, other.cycle_type))
            )
            # total < 0 => c2's cycles are "later" ie. has terms with many elems
            if total < 0:
                return True
        # prefer terms that are closer to ascending in order
        if str(self) < str(other):
            return True  # ie. (1 2 3 4) should be preferred over (1 3 2 4)

        return False
