"""
Top-level Group class for working with mathematical groups.

It includes functionality for generating group elements,
checking subgroup properties, and performing group operations.
"""

from collections.abc import Iterable
from typing import List
from tqdm import tqdm

from . import utils
from .group_element import GroupElement


class Group(set):
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

    def _parse(self, equation: str, initial=False) -> GroupElement:  # helper function for creating new expressions
        raise NotImplementedError

    def _generate_all(self):
        # generate all elements in the group
        raise NotImplementedError

    def _same_group_type(self, other: "Group"):
        # check if 2 Groups are subgroups of the same group
        return type(self) is type(other)

    def _identity_expr(self):
          # helper function to return an expr, pretty much only used in generate
        raise NotImplementedError

    def _identity_group(self):   # helper function return a Group containing only an identity {Expression, Permutation}
        expr = self._identity_expr()
        return self.subgroup(expr)

    def generators(self) -> List[GroupElement]:
        raise NotImplementedError

    def subgroup(self, *elems):  # create an empty subgroup that has the same multiplication rules
        group = Group(*elems, name=self.name, verbose=self.verbose)
        set_these = ["singleton_rules", "general_rules", "n", "symbols",
                     "simplify_cache", "quotient_map"]
        for var_name in set_these:
            if hasattr(self, var_name):
                obj = getattr(self, var_name)
                if var_name in ['quotient_map'] and obj:
                    obj = obj.copy()
                setattr(group, var_name, obj)
        return group

    def evaluate(self, equation):
        if isinstance(equation, str):
            return self._parse(equation).simplify()
        elif isinstance(equation, (list,tuple)):
            return [self.evaluate(s) for s in equation]  # => recursive
        else: # ie. Expression, Permutation, Term
            return equation

    def copy(self):
        return self.subgroup(*self)

    def __iter__(self, track=False):
        if not self.has_elems:
            print("Warning: you are trying to iterate over an empty group")
        iterator = super().__iter__()
        if track:
            return tqdm(iterator)
        return iterator


    # Properties


    @property
    def is_perm_group(self):
        return self.n is not None  # permutation groups have n defined, others don't

    @property
    def has_elems(self):
        return len(self) > 0

    def is_subgroup(self, verbose=True):
        if not self.has_elems:
            return False
        for elem1 in self:
            for elem2 in self:
                if elem1/elem2 not in self:
                    if verbose:
                        print(f"{elem1=}, {elem2=} generates {elem1/elem2} not in subgroup")
                    return False
        return True

    def is_normal(self, subgroup, verbose=False):
        if not subgroup.is_subgroup(verbose=verbose):
            if verbose:
                print("not even a subgroup")
            return False
        for h in subgroup:
            for g in self:
                if g * h / g not in subgroup:
                    if verbose:
                        print(f"group_elem={g}, subgroup_elem={h} generates {g*h/g} not in subgroup")
                    return False
        return True


    # Operations


    def __mul__(self, other):  # ie Group * [Expression, Group, str, list[str]] (right cosets)
        if isinstance(other, (symbolic.Expression, permutation.Permutation)):
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


    def __rmul__(self, other): # ie. Expression * Group (left cosets)
        if isinstance(other, (symbolic.Expression, permutation.Permutation)):
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


    def __truediv__(self, other): # ie. Group / {Term, Permutation}
        if isinstance(other, Group):
            if not self._same_group_type(other):
                raise ValueError("Incompatible group types {self.name} and {other.name}")
            if not self.is_normal(other):
                raise ValueError("Attempting to quotient by a non-normal subgroup")
            cosets = self.find_cosets(other)
            # print("cosets", cosets)
            quotient_map = {x: representative for representative, coset in cosets.items() for x in coset}
            reprs = cosets.keys()
            quotient = self.subgroup(*reprs)
            for representative in quotient_map.values():
                representative.group = quotient  # update the group in the map to the new, correct thing
            quotient.quotient_map = quotient_map
            return quotient

        elif isinstance(other, (symbolic.Expression, permutation.Permutation)):
            return self*other.inv()
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
                next = start*elem
                if next not in visited:
                    # print("found new node", next)
                    frontier.add(next)
                    visited.add(next)
                    #yield next  # so that we can do infinite groups as well
        return visited


    def centralizer(self, elems):
        elems = self.evaluate(elems)
        if not isinstance(elems, Iterable):
            elems = [elems]
        commuters = self.subgroup()
        for candidate in self:
            for pt in elems:
                if pt*candidate != candidate*pt:
                    break
            else:
                commuters.add(candidate)
        return commuters


    def center(self):
        return self.centralizer(self)


    def conjugacy_class(self, elem, paired=False, track=False):
        reachable = []
        generators = []  # the associated list of elements that generate each coset/element in "reachable"
        elem = self.evaluate(elem)
        for other in self.__iter__(track=track):
            new_elem = other * elem / other
            #print(other, "generates", new_elem)
            if new_elem not in reachable:
                reachable.append(new_elem)
                generators.append(other)
            elif paired:  # if we want to know what to conjugate with to get each element in the conj_class,
                idx = reachable.index(new_elem) # then set paired=True. This bit just picks the 'simplest' such element
                if other.simpler_heuristic(generators[idx]):
                    generators[idx] = other
        if paired:
            return dict(zip(generators, reachable))
        elif not isinstance(elem, Group):
            return self.subgroup(*reachable)
        else:
            return reachable


    def orbit(self, base_elem):
        base_elem = self.evaluate(base_elem)

        reachable = self.subgroup()
        elem = base_elem
        reachable.add(elem)
        while not elem.is_identity:
            elem = elem*base_elem
            reachable.add(elem)
        return reachable


    def normalizer(self, elems): # no need to do .evaluate here, since we .generate anyway
        if not isinstance(elems, Group):
            elems = self.generate(elems)
        commuters = self.subgroup()
        for candidate in self:
            for elem in elems:
                if candidate*elem/candidate not in elems:
                    break
            else:
                commuters.add(candidate)
        return commuters

    def normal_closure(self, elems):  # return smallest normal subgroup that contains `elems`
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
                best_representative = elem  # heuristically find the simplest representative
                for representative in new_coset:
                    if representative.simpler_heuristic(best_representative):
                        best_representative = representative
                cosets[best_representative] = new_coset
                full_group = full_group - new_coset
        return cosets

    def commutator(self, track=False):
        elems = self.subgroup()
        for x in self.__iter__(track=track):
            for y in self:
                elems.add(x * y / x / y)
        return self.generate(elems)
