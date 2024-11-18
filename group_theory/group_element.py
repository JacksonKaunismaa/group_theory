from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

T = TypeVar("T", bound="GroupElement")

class GroupElement(ABC, Generic[T]):
    @abstractmethod
    def _parse(self, equation: str, initial: bool) -> Any:
        ...

    @abstractmethod
    def simplify(self) -> "GroupElement":
        ...

    @abstractmethod
    def inv(self) -> "GroupElement":
        ...

    @abstractmethod
    def __mul__(self, other: "GroupElement") -> "GroupElement":
        ...

    @abstractmethod
    def __truediv__(self, other: "GroupElement") -> "GroupElement":
        ...

    @abstractmethod
    def simpler_heuristic(self, other: T) -> bool:
        # returns True if term1 is heuristically "simpler" than term2
        if self.is_identity or other.is_identity:
            return self.is_identity
