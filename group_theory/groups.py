"""
Top-level Group class for working with mathematical groups.

It includes functionality for generating group elements,
checking subgroup properties, and performing group operations.
"""

from collections.abc import Iterable
from typing import Any, Generic, List, Union
from tqdm import tqdm

from .group_element import GroupElement, T


class Group(set, Generic[T]):
    """
    Represents a mathematical group, including both the elements of the group
    and the rules of the representation.

    Attributes:
        singleton_rules (dict): Maps symbols to their orbit size (for symbolic groups).
        general_rules (defaultdict): General rules for group operations.
        symbols (set): Set of symbols used in the group.
        simplify_cache (dict): Cache for speeding up simplification.
        verbose (bool): Verbosity flag.
        name (str): Name of the group.
        n (int): Order of the group (for permutation groups).
        quotient_map (dict): Map of elements to the simplest representative for quotient groups.

    Methods:
        subgroup(*elems): Creates a subgroup with the same multiplication rules.
        evaluate(str): Evaluates an equation and simplifies it.
        copy(): Returns a copy of the group.
        generate(*exprs): Generates a group from given expressions.
        centralizer(elems): Returns the centralizer of given elements.
        center(): Returns the center of the group.
        conjugacy_class(elem, paired=False, track=False): Returns the conjugacy class of an element.
        orbit(base_elem): Returns the orbit of a base element.
        normalizer(elems): Returns the normalizer of given elements.
        normal_closure(elems): Returns the smallest normal subgroup containing given elements.
        normal_core(elems): Returns the largest normal subgroup contained in given elements.
        find_cosets(coset, left=True): Finds the cosets of a subgroup.
        commutator(track=False): Returns the commutator subgroup.
    """

    def __init__(self, *elems: GroupElement, name: str, verbose: bool):
        super().__init__(elems)
        self.verbose = verbose
        self.name = name
        self.quotient_map = None

    def _parse(
        self, equation: Any, initial=False
    ) -> GroupElement:  # helper function for creating new expressions
        raise NotImplementedError

    def _generate_all(self):
        # generate all elements in the group
        raise NotImplementedError

    def _same_group_type(self, other: "Group"):
        # check if 2 Groups are subgroups of the same group
        return type(self) is type(other)

    def identity_expr(self):
        # helper function to return an expr, pretty much only used in generate
        raise NotImplementedError

    def _identity_group(
        self,
    ):  # helper function return a Group containing only an identity {Expression, Permutation}
        expr = self.identity_expr()
        return self.subgroup(expr)

    def generators(self) -> List[T]:
        raise NotImplementedError

    def copy_subgroup_attrs_to(self, subgroup: "Group"):
        """Copy important attrs to another group for subgroup creation"""
        if self.quotient_map is not None:
            subgroup.quotient_map = self.quotient_map.copy()

    def subgroup(self, *elems: T) -> "Group":
        raise NotImplementedError

    def evaluate(self, equation: Union[T, Any]) -> T:
        if isinstance(equation, GroupElement):
            return equation  # type: ignore
        return self._parse(equation).simplify()

    def evaluates(self, *equations: Any | T) -> List[T]:
        return [self.evaluate(eq) for eq in equations]

    def copy(self):
        return self.subgroup(*self)

    def iterate(self, track=False):
        if not self.has_elems:
            print("Warning: you are trying to iterate over an empty group")
        iterator = super().__iter__()
        return iter(tqdm(iterator, disable=not track))

    def __iter__(self):
        return self.iterate()

    # Properties

    @property
    def has_elems(self):
        return len(self) > 0

    def is_subgroup(self):
        if not self.has_elems:
            return False
        for elem1 in self:
            for elem2 in self:
                if elem1 / elem2 not in self:
                    if self.verbose:
                        print(
                            f"{elem1=}, {elem2=} generates {elem1/elem2} not in subgroup"
                        )
                    return False
        return True

    def is_normal(self, subgroup: "Group"):
        if not subgroup.is_subgroup():
            if self.verbose:
                print("not even a subgroup")
            return False
        for h in subgroup:
            for g in self:
                if g * h / g not in subgroup:
                    if self.verbose:
                        print(
                            f"group_elem={g}, subgroup_elem={h} generates {g*h/g} not in subgroup"
                        )
                    return False
        return True

    # Operations

    def __mul__(self, other: Union[GroupElement, "Group", str, List[str]]):
        # ie Group * [Expression, Group, str, list[str]] (right cosets)
        if isinstance(other, GroupElement):
            new_elems = self.subgroup()
            for elem in self:
                new_elems.add(elem * other)
            return new_elems
        elif isinstance(other, Group) and self._same_group_type(other):
            new_elems = self.subgroup()
            for e1 in self:
                for e2 in other:
                    new_elems.add(e1 * e2)
            return new_elems
        elif isinstance(other, str):
            elem = self.evaluate(other)
            return self * elem
        elif isinstance(other, list) and isinstance(other[0], str):
            elems = self.generate(*other)
            return self * elems
        else:
            return NotImplemented

    def __rmul__(self, other):  # ie. Expression * Group (left cosets)
        if isinstance(other, GroupElement):
            new_elems = self.subgroup()
            for elem in self:
                new_elems.add(other * elem)
            return new_elems
        elif isinstance(other, str):
            elem = self.evaluate(other)
            return elem * self
        elif isinstance(other, list) and isinstance(other[0], str):
            elems = self.generate(*other)
            return elems * self
        else:
            return NotImplemented

    def __truediv__(self, other):  # ie. Group / {Term, Permutation}
        if isinstance(other, Group):
            if not self._same_group_type(other):
                raise ValueError(
                    "Incompatible group types {self.name} and {other.name}"
                )
            if not self.is_normal(other):
                raise ValueError("Attempting to quotient by a non-normal subgroup")
            cosets = self.find_cosets(other)
            # print("cosets", cosets)
            quotient_map = {
                x: representative
                for representative, coset in cosets.items()
                for x in coset
            }
            reprs = cosets.keys()
            quotient = self.subgroup(*reprs)
            for representative in quotient_map.values():
                representative.group = (
                    quotient  # update the group in the map to the new, correct thing
                )
            quotient.quotient_map = quotient_map
            return quotient

        elif isinstance(other, GroupElement):
            return self * other.inv()
        elif isinstance(other, list) and isinstance(other[0], str):
            elems = self.generate(*other)
            return self / elems
        else:
            return NotImplemented

    def __and__(self, other):  # make sure to cast to a Group object
        return self.subgroup(*super().__and__(other))

    def __or__(self, other):
        return self.subgroup(*super().__or__(other))

    def generate(self, *exprs) -> "Group":
        if len(exprs) == 0:
            return self._identity_group()

        flat_exprs = set()
        for expr in exprs:
            if isinstance(expr, str):
                flat_exprs.add(self.evaluate(expr))
            elif isinstance(expr, Group):
                flat_exprs |= expr
            else:
                raise ValueError(f"Unknown type '{type(expr)}' in exprs list")

        frontier = self.subgroup(*flat_exprs)
        visited = self.subgroup()
        # print("frontier", frontier)
        while len(frontier) > 0:  # BFS
            start = frontier.pop()
            # print("checking elem", start)
            for elem in flat_exprs:
                next_elem = start * elem
                if next_elem not in visited:
                    # print("found new node", next)
                    frontier.add(next_elem)
                    visited.add(next_elem)
                    # yield next  # so that we can do infinite groups as well
        return visited

    def centralizer(self, elems: Union["Group", List[Any]]) -> "Group":
        if isinstance(elems, list):
            elems = self.evaluates(*elems)

        if not isinstance(elems, Iterable):
            elems = [elems]
        commuters = self.subgroup()
        for candidate in self:
            for pt in elems:
                if pt * candidate != candidate * pt:
                    break
            else:
                commuters.add(candidate)
        return commuters

    def center(self):
        return self.centralizer(self)

    def conjugacy_class(self, elem, paired=False, track=False):
        reachable = []
        generators = (
            []
        )  # the associated list of elements that generate each coset/element in "reachable"
        elem = self.evaluate(elem)
        for other in self.iterate(track=track):
            new_elem = other * elem / other
            # print(other, "generates", new_elem)
            if new_elem not in reachable:
                reachable.append(new_elem)
                generators.append(other)
            elif (
                paired
            ):  # if we want to know what to conjugate with to get each element in the conj_class,
                idx = reachable.index(
                    new_elem
                )  # then set paired=True. This bit just picks the 'simplest' such element
                if other.simpler_heuristic(generators[idx]):
                    generators[idx] = other
        if paired:
            return dict(zip(generators, reachable))
        elif not isinstance(elem, Group):
            return self.subgroup(*reachable)
        else:
            return reachable

    def orbit(self, base_elem: GroupElement):
        base_elem = self.evaluate(base_elem)

        reachable = self.subgroup()
        elem = base_elem
        reachable.add(elem)
        while not elem.is_identity:
            elem = elem * base_elem
            reachable.add(elem)
        return reachable

    def normalizer(
        self, elems
    ):  # no need to do .evaluate here, since we .generate anyway
        if not isinstance(elems, Group):
            elems = self.generate(elems)
        commuters = self.subgroup()
        for candidate in self:
            for elem in elems:
                if candidate * elem / candidate not in elems:
                    break
            else:
                commuters.add(candidate)
        return commuters

    def normal_closure(
        self, elems
    ):  # return smallest normal subgroup that contains `elems`
        if not isinstance(elems, Iterable):
            elems = self.generate(elems)
        expanded = self.subgroup()
        for g in self:
            expanded |= g * elems / g
        # print(expanded, "expanded")
        return self.generate(expanded)

    def normal_core(self, elems):  # return largest normal subgroup contained in `elems`
        if not isinstance(elems, Iterable):
            elems = self.generate(elems)
        expanded = self.subgroup(*elems)
        for g in self:
            expanded &= g * elems / g
        return expanded

    def find_cosets(self, coset: "Group", left=True):
        cosets = {}
        full_group = self.copy()
        while len(full_group) > 0:
            elem = full_group.pop()
            if left:
                new_coset = elem * coset
            else:
                new_coset = coset * elem
            if new_coset not in cosets.values():
                best_representative = (
                    elem  # heuristically find the simplest representative
                )
                for representative in new_coset:
                    if representative.simpler_heuristic(best_representative):
                        best_representative = representative
                cosets[best_representative] = new_coset
                full_group = full_group - new_coset
        return cosets

    def commutator(self, track=False):
        elems = self.subgroup()
        for x in self.iterate(track=track):
            for y in self:
                elems.add(x * y / x / y)
        return self.generate(elems)
