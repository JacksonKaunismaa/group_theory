from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

T = TypeVar("T", bound="GroupElement")


class GroupElement(ABC, Generic[T]):
    """
    Abstract base class representing a group element in group theory.

    Methods
    -------
    _parse(equation: str, initial: bool) -> Any
        Parse a given equation string.
    simplify() -> "GroupElement"
        Simplify the group element.
    inv() -> "GroupElement"
        Return the inverse of the group element.
    __mul__(other: T) -> "GroupElement"
        Define the multiplication operation with another group element.
    __truediv__(other: T) -> "GroupElement"
        Define the division operation with another group element.
    is_identity -> bool
        Check if the group element is the identity element.
    simpler_heuristic(other: T) -> bool
        Determine if the current element is heuristically simpler than another element.
    """

    @abstractmethod
    def _parse(self, equation: str, initial: bool) -> Any: ...

    @abstractmethod
    def simplify(self) -> T: ...

    @abstractmethod
    def inv(self) -> "GroupElement": ...

    @abstractmethod
    def __mul__(self, other: T) -> "GroupElement": ...

    @abstractmethod
    def __truediv__(self, other: T) -> "GroupElement": ...

    @property
    @abstractmethod
    def is_identity(self) -> bool: ...

    @abstractmethod
    def simpler_heuristic(self, other: T) -> bool:
        """Returns True if term1 is heuristically "simpler" than term2"""
        if self.is_identity or other.is_identity:
            return self.is_identity
