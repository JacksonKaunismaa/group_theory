from collections import defaultdict
from typing import Union, List, Optional

# import copy

from . import utils
from .group_element import GroupElement
from .groups import Group


IDENTITY_SYMBOLS = ["e", "1"]


class SymbolicGroup(Group):
    def __init__(
        self,
        *elems: "Expression",
        rules: List[str],
        name: str = "",
        generate: bool = False,
        verbose: bool = False,
    ):
        super().__init__(*elems, name=name, verbose=verbose)
        self.singleton_rules = {}
        self.general_rules = defaultdict(list)
        self.symbols = set()
        self.simplify_cache = {}

        # parse rules
        for rule in rules:
            pattern, result = rule.split("=")
            pattern_expr = self._parse(pattern, initial=True)
            result_expr = self._parse(result, initial=True)

            self._add_syms(
                pattern_expr, result_expr
            )  # so that we can generate the group later

            if (
                len(pattern_expr) == 1 and result_expr.is_identity
            ):  # if symbol is cyclic, do this for efficiency
                self.singleton_rules[pattern_expr[0].sym] = pattern_expr[0].exp
                continue
            self.general_rules[len(pattern_expr)].append(
                (pattern_expr, result_expr)
            )  # map symbol -> (exponent, replacement)

        if generate:
            self._generate_all()

    def _add_syms(self, *exprs: "Expression"):
        for expr in exprs:
            for term in expr:
                if not term.is_identity:
                    self.symbols.add(term.sym)

    def _generate_all(self):
        elems = self.generate(*self.symbols)
        self |= elems

    def _same_group_type(self, other: Group):
        types_match = super()._same_group_type(other)
        if types_match:
            return (
                self.general_rules == other.general_rules
                and self.singleton_rules == other.singleton_rules
            )
        return False

    def identity_expr(self):
        return Expression([Term.identity()], self)

    def _parse(self, equation: str, initial: bool = False) -> "Expression":
        return Expression(equation, self, initial=initial)

    def generators(self):
        return list(self.symbols)


class Term(
    GroupElement
):  # a single instance of something like "r^3", r is the sym, 3 is the exp
    def __init__(
        self,
        sym: str | None,
        exp: int,
        group: Optional["Group"] = None,
        cyclic_rule: Optional[int] = None,
    ):
        # cylcic rule is the number of times the symbol can be repeated before it wraps around
        # can specify either a Group, which is only used to extract the cyclic_rule, or pass cyclic_rule directly
        self.sym = sym
        self.exp = exp
        self.cyclic_rule = cyclic_rule
        if group is not None:
            self.cyclic_rule = group.singleton_rules.get(
                sym, None
            )  # mostly for efficiency
        self.simplify()

    def simplify(self):
        if self.cyclic_rule:
            if self.exp < 0:
                self.exp += self.cyclic_rule * (1 + (-self.exp) // self.cyclic_rule)
            self.exp = self.exp % self.cyclic_rule
        if self.exp == 0:
            self.sym = None  # make self identity, not great since now we have 2 implementations of identity() (other is in Group)
        return self

    def _parse(self, equation: str, initial: bool):
        pass

    @staticmethod
    def identity():
        return Term(None, 0)

    @property
    def is_identity(self):
        return self.sym is None

    # LHS takes precendence, for Term * {Term, Expression}, backend multiplication to match the Expression API
    # Expression is returned if its Term*Expression
    # Term is returned if its Term*Term and the terms could be combined
    # List[Term] is returned if its Term*Term and the terms couldn't be combined
    def _mul(self, other) -> Union["Expression", "Term", List["Term"]]:
        if self.is_identity:
            return other

        if isinstance(other, Term):  # Term * Term multiplication
            if other.is_identity:  # to avoid NoneType issues
                return self
            if self.sym == other.sym:
                return Term(
                    self.sym, self.exp + other.exp, cyclic_rule=self.cyclic_rule
                )
            else:
                return [self, other]

        elif isinstance(other, Expression):  # Term * Expression multiplication
            return Expression([self], other.group)._mul(other)
        else:
            print(type(other), "type")
            raise NotImplementedError(
                f"Don't know how to multiply Term * {type(other)}"
            )

    # def copy(self):
    #     return Term(self.sym, self.exp, cyclic_rule=self.cyclic_rule)

    # frontend of multiplication
    def __mul__(self, other: GroupElement) -> Union["Expression", "Term", List["Term"]]:
        if isinstance(other, (Expression, Term)):
            return self._mul(other)
        else:
            return NotImplemented

    def __repr__(self):
        if self.exp == 1:
            return str(self.sym)
        if self.is_identity:
            return "e"
        return f"{self.sym}{self.exp}"

    # def __len__(self):  # to be consistent with Expression, but shouldn't ever get called
    #     return 1

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.sym == other.sym and self.exp == other.exp

    def __ge__(self, other):
        if self.is_identity:
            return other.is_identity
        return self.sym == other.sym and self.exp >= other.exp

    def inv(self):
        if self.is_identity:
            return self
        return Term(self.sym, -self.exp, cyclic_rule=self.cyclic_rule)

    # def __pow__(self, other):
    #     return Term(self.sym, self.exp*other, cyclic_rule=self.cyclic_rule).simplify()

    # backend of division (no simplify step)
    def _truediv(self, other):
        return self._mul(other.inv())

    # frontend division (yes simplify step)
    def __truediv__(self, other):
        if isinstance(other, (Expression, Term)):
            return self._mul(other.inv()).simplify()
        else:
            return NotImplemented

    def simpler_heuristic(self, other: "Term") -> bool:
        identity_check = super().simpler_heuristic(other)
        if identity_check is not None:
            return identity_check
        return self.exp < other.exp


class Expression(GroupElement):
    """
    Represents an element of a symbolic group, as a sequence of terms.

    Attributes:
        group (SymbolicGroup): The group to which the expression belongs.
        expr (List[Term]): The list of terms in the expression.

    Methods:
        __init__(expr, group, initial=False):
            Initializes an Expression object.

        _parse(equation, initial):
            Parses a string equation into a list of Term objects.

        windows_match(window, pattern):
            Checks if a window of terms matches a given pattern.

        tprint(*args, **kwargs):
            Prints messages if the group's verbose attribute is True.

        simplify(max_iters=200):
            Simplifies the expression by applying group rules.

        _combine_like_terms(n):
            Combines like terms in the expression starting from a given position.

        _filter_identity(expr=None):
            Removes identity terms from the expression or a list of terms.

        is_identity:
            Checks if the expression is an identity expression.

        __getitem__(idx):
            Gets the term at the specified index.

        __len__():
            Returns the number of terms in the expression.

        _concat(left, right):
            Concatenates two expressions or lists of terms without combining like terms.

        _mul(other):
            Multiplies the expression with another expression, term, or list of terms.

        __mul__(other):
            Multiplies the expression with another expression or term and simplifies the result.

        inv():
            Returns the inverse of the expression.

        _truediv(other):
            Divides the expression by another expression or term without simplifying.

        __truediv__(other):
            Divides the expression by another expression or term and simplifies the result.

        __pow__(other):
            Raises the expression to the power of an integer.

        __eq__(other):
            Checks if two expressions are equal.

        __repr__():
            Returns a string representation of the expression.

        __hash__():
            Returns the hash of the expression.

        simpler_heuristic(other):
            Determines if the current expression is simpler than another expression based on heuristic rules.
    """

    def __init__(
        self, expr: Union[List[Term], str], group: "SymbolicGroup", initial=False
    ):
        self.group = group
        if isinstance(expr, str):  # parse it if its a string
            expr = self._parse(expr, initial=initial)
        self.expr = expr
        if not self.expr:
            self.expr = [
                Term.identity()
            ]  # allowing empty expresions makes things buggy

    def _parse(self, equation: str, initial: bool) -> List[Term]:
        # initial=True when parsing the group rules themselves, since we don't know
        # 'symbols' at that point, so we can't check against it
        terms = equation.strip().split()
        start = self.group.identity_expr()
        for t in terms:
            if t[0] in IDENTITY_SYMBOLS:
                next_term = self.group.identity_expr()
            elif t[0] not in self.group.symbols and not initial:
                raise ValueError(
                    f"Unknown symbol '{t[0]}' while parsing expression of a '{self.group.name}' group."
                    f"Possible symbols are '{self.group.symbols}'"
                )
            elif len(t) == 1:
                next_term = Term(t[0], 1, group=self.group)  # 1 is default exponent
            else:
                next_term = Term(t[0], int(t[1:]), group=self.group)
            start = start._mul(next_term)
        if isinstance(start, Term):  # always return List[Term]
            return [start]
        return start.expr

    def windows_match(self, window, pattern):
        if (window[0] >= pattern[0]) and (window[-1] >= pattern[-1]):
            for actual, expected in zip(window[1:-1], pattern[1:-1]):
                if actual != expected:
                    return False
            return True
        return False

    def tprint(self, *args, **kwargs):
        if self.group.verbose:
            print(*args, **kwargs)

    def simplify(self, max_iters=200) -> "Expression":
        updated = True  # if we've applied a rule or not in the given iteration
        n = 0  # check total iterations so far
        # self.tprint("doing the printing?")
        simplified = self  # working copy that we will be writing to

        while updated and n < max_iters:
            n += 1
            # self.tprint("curr expr is", simplified)
            updated = False
            # simplified = simplified._filter_identity()  # first step of simplificiation is eliminating identities

            if simplified in self.group.simplify_cache:  # try exiting early
                # self.tprint("found in cache", simplified)
                simplified.expr = self.group.simplify_cache[simplified]
                break

            for window_size in sorted(
                self.group.general_rules.keys(), reverse=True
            ):  # check all possible window-sizes we have rules for
                if len(simplified) < window_size:
                    continue
                new_expr = self.group.identity_expr()

                window_iter = utils.sliding_window(
                    simplified.expr, window_size
                )  # current window we are looking to apply rules in
                last_posn = 0  # track where we've appended up to, so we can append missing ones at the end
                for window in window_iter:
                    # self.tprint("checking window of", window, type(window))
                    for pattern, result in self.group.general_rules[
                        window_size
                    ]:  # check all possible rules at this given window
                        # self.tprint("\tchecking window against", pattern)
                        if self.windows_match(window, pattern):  # if the rule applies
                            # self.tprint("\t\twindow matches, proceeding with replacement...")
                            updated = True
                            # self.tprint("\t\tbefore replacing, new_expr is now", type(new_expr), new_expr)

                            # the result of applying the rule.
                            # filter identity to save a bit of compute (calculating translation doesn't filter for it
                            # ie. window[0]/pattern[0] * result * window[-1]/pattern[-1]

                            if window_size == 2 and len(result) == 2:
                                # very specific optimization for special type of rules
                                # based on the idea that if `s r = r^k s`, then `s^m r^l = r^(l*k^m) s^m`
                                left_exponent = window[0].exp  # m
                                right_exponent = window[1].exp  # l
                                result_exp = result[0].exp  # k
                                # do modular exponentiation for a speed-up
                                new_exp = right_exponent * pow(
                                    result_exp, left_exponent, result[0].cyclic_rule
                                )  #
                                translation = Expression(
                                    [
                                        Term(
                                            result[0].sym,
                                            new_exp,
                                            cyclic_rule=result[0].cyclic_rule,
                                        ),
                                        Term(
                                            result[1].sym,
                                            left_exponent,
                                            cyclic_rule=result[1].cyclic_rule,
                                        ),
                                    ],
                                    self.group,
                                )
                            elif window_size > 1:
                                translation = result._concat(
                                    [window[0]._truediv(pattern[0])],
                                    [window[-1]._truediv(pattern[-1])],
                                )
                            else:  # window_size == 1
                                translation = result._concat(
                                    [], [window[-1]._truediv(pattern[-1])]
                                )

                            # self.tprint("\t\twill be adding", translation)

                            new_expr = new_expr._mul(translation)
                            # self.tprint("\t\tafter replacing, new_expr is now", new_expr)
                            # self.tprint("\t\twindow matched, advancing window by", window_size, "spaces")
                            try:  # skip window_size worth of windows because we've already used all those terms
                                last_posn += window_size
                                # self.tprint("\t\tnew last posn", last_posn)
                                for _ in range(window_size - 1):
                                    window = next(window_iter)
                                # window = [next(window_iter) for _ in range(window_size)][0]
                            except StopIteration:
                                # self.tprint("stop iter")
                                break  # no need to check other rules if we've ran out of windows to look at
                            # self.tprint("\t\tafter advancing, new_expr is now", new_expr)
                            break  # we've made a valid pattern match at this location, so don't check any more patterns
                    else:  # if we reach the end, we've checked all patterns and nothing worked, so just append 1 term and move the window
                        # self.tprint("\tbefore appending, new_expr is now", new_expr)
                        new_expr = new_expr._mul(window[0])
                        last_posn += 1
                        # self.tprint("\tafter appending, new_expr is now", new_expr)

                # self.tprint("appending last window of", simplified[last_posn:])
                if last_posn != len(
                    simplified
                ):  # append any missing terms that got skipped over because we moved window
                    new_expr = new_expr._mul(simplified[last_posn:])
                # self.tprint("end_cycle new_expr is now", new_expr)
                simplified = new_expr
        # self.tprint("n was", n, max_iters)
        # self.tprint(simplified.group, self.group)
        self.group.simplify_cache[self] = simplified.expr
        # technically there should be .copy() on each return statement, but unless the client is doing something like
        # directly modifying .exp or .sym fields, it won't really have problems, since every "proper" operation will
        # create new Terms/Expressions as needed and won't mess up the simplify_cache
        if (
            self.group.quotient_map
        ):  # don't bother checking existence, since that should throw an error anyway
            return self.group.quotient_map[simplified]  # .copy()
        return simplified  # .copy()

    def combine_like_terms(self, n: int) -> "Expression":
        """
        Combine like terms in the expression starting from the specified index.

        This method attempts to combine like terms in the expression starting from the 'gap'
        between the terms at index `n-1` and `n`, and works its way outwards. The combination
        process involves checking if adjacent terms can be combined into a single term. If
        they can be combined, the method continues to the next pair of terms. If not, it stops
        and concatenates the combined terms with the remaining terms of the expression.

        Args:
            n (int): The index from which to start combining like terms.

        Returns:
            Expression: A new expression with like terms combined, or the modified expression
            if it was updated in place.
        """
        # if self is abcd * xyz then we should combine like terms starting from the 'gap' between
        # d and x, and work our way out. So check if d*x can be combined as like terms, if yes,
        # then see if c*dx*y can be combined, and so on. We detect the 'if they can be combined' by seeing if the
        # result of multiplying them (which should be a Term*Term multiplication in each case) actually results in List[Term]
        # which would imply that the multiply was multiplying 2 Terms with different base symbols and so was forced
        # to concatenate them into a List[Term] object
        curr_term = Term.identity()
        for i, (term1, term2) in enumerate(zip(self.expr[n - 1 :: -1], self.expr[n:])):
            # print("looking at", term1, curr_term, term2)
            curr_term = term1._mul(curr_term)  #  * term2
            curr_term = curr_term._mul(term2)
            if isinstance(
                curr_term, list
            ):  # in Term*Term multiplication list => Terms couldn't be combined
                break
        else:  # at this point curr_term will be a Term, since we didn't break out
            curr_term = [curr_term]
        curr_term = self.filter_identity(curr_term)
        new_expr = curr_term._concat(
            self.expr[: n - i - 1], self.expr[n + i + 1 :]
        )  # self.expr[:n-i-1] * curr_term *  self.expr[n+i+1:]

        if isinstance(new_expr, Expression):
            return new_expr
        self.expr = new_expr  # can modify in-place here since only used via __mul__, which creates a new Expression anyway
        return self

    def filter_identity(
        self, expr=None
    ) -> "Expression":  # remove all identity terms from an Expression or a list
        if expr is None:  # need the expr parameter since sometimes it handles lists
            expr = self.expr
        if isinstance(expr, Expression):
            expr_terms = expr.expr
        else:
            expr_terms = expr
        return Expression(
            list(filter(lambda x: not x.is_identity, expr_terms)), self.group
        )

    @property
    def is_identity(self):
        return all([x.is_identity for x in self.expr])

    def __getitem__(self, idx) -> Term:
        return self.expr[idx]

    def __len__(self):
        return len(self.expr)

    # the pupose of this methods is to multiply 2 Expressions or lists such that combine_like_terms
    # isn't called. Used to avoid infinite loops, but probably shouldn't be used by clients. Should only use
    # when you know the combining like terms won't do anything
    # for multiplying {list,Expression} * Expression * {list, Expression}
    def _concat(
        self, left: Union[list, "Expression"], right: Union[list, "Expression"]
    ) -> "Expression":
        if isinstance(left, list):
            left = Expression(left, self.group)
        if isinstance(right, list):
            right = Expression(right, self.group)
        return Expression(
            left.expr + self.expr + right.expr, self.group
        ).filter_identity()

    # def copy(self):  # need deepcopy since its a list
    #     # return Expression(copy.deepcopy(self.expr), self.group)
    #     return Expression(self.expr.copy(), self.group)

    def _mul(self, other) -> "Expression":  # for Expression * {Expression, Term}
        """Combine expressions without calling .simplify().

        This is used to avoid infinite recursion when multiplying expressions.
        Still combines like terms between the end of self and the start of other,
        but doesn't simplify the result.
        """
        if isinstance(other, Expression):
            other_expr = other.expr
        elif isinstance(other, list):
            other_expr = other
        elif isinstance(other, Term):
            other_expr = [other]
        else:
            raise NotImplementedError(
                f"Don't know how to multiply Expression * {type(other)}"
            )
        new_expr = Expression(self.expr + other_expr, self.group)
        return new_expr.combine_like_terms(len(self))

    # frontend of multiplication, so that simplification is done automatically
    def __mul__(self, other):  # Expression * {Expression, Term}
        if isinstance(other, (Expression, Term)):
            return self._mul(other).simplify()
        else:
            return NotImplemented

    def inv(self):
        if self.is_identity:
            return self
        return Expression([x.inv() for x in self.expr[::-1]], self.group)

    # backend of division (no simplify step)
    def _truediv(self, other):
        return self._mul(other.inv())

    # frontend division (with simplify step)
    def __truediv__(self, other):
        if isinstance(other, (Term, Expression)):
            return self._mul(other.inv()).simplify()
        else:
            return NotImplemented

    def __pow__(self, other):
        curr_expr = self.group.identity_expr()
        for _ in range(other):
            curr_expr *= self
        return curr_expr

    def __eq__(self, other):  # need to check len because zip truncates elements
        try:
            return len(self) == len(other) and all(
                t1 == t2 for t1, t2 in zip(self.expr, other.expr)
            )
        except TypeError:
            print(self, type(self), "######", other, type(other))
            raise

    def __repr__(self):
        return " ".join([str(t) for t in self.expr])

    def __hash__(self):
        return hash(str(self))

    def simpler_heuristic(self, other: "Expression") -> bool:
        identity_check = super().simpler_heuristic(other)
        if identity_check is not None:
            return identity_check

        if len(self) < len(other):  # shorter Expressions are better
            return True
        if sum(x.exp for x in self) < sum(
            x.exp for x in other
        ):  # smaller exponents are better
            return True
        return False
