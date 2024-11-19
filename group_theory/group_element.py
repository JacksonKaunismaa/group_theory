from abc import ABC, abstractmethod
from typing import List, Self


class GroupElement(ABC):
    """
    Abstract base class representing a group element in group theory.

    Methods
    -------
    _parse(equation: str, initial: bool) -> Any
        Parse a given equation string.
    simplify() -> Self
        Simplify the group element.
    inv() -> Self
        Return the inverse of the group element.
    __mul__(other: Self) -> Self
        Define the multiplication operation with another group element.
    __truediv__(other: Self) -> Self
        Define the division operation with another group element.
    is_identity -> bool
        Check if the group element is the identity element.
    simpler_heuristic(other: Self) -> bool
        Determine if the current element is heuristically simpler than another element.
    """

    @abstractmethod
    def _parse(self, equation: str, initial: bool) -> List: ...

    @abstractmethod
    def simplify(self) -> Self: ...

    @abstractmethod
    def inv(self) -> Self: ...

    @abstractmethod
    def __mul__(self, other: Self) -> Self: ...

    @abstractmethod
    def __truediv__(self, other: Self) -> Self: ...

    @property
    @abstractmethod
    def is_identity(self) -> bool: ...

    @abstractmethod
    def simpler_heuristic(self, other: Self) -> bool: ...
