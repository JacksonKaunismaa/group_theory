# Group Theory

A Python package for performing computations in group theory. It includes implementations of permutation groups, symbolic groups, and utilities for working with group elements.

## Project Structure

- `group_theory/`
    - `__init__.py`: Initialization file for the group_theory module.
    - `groups.py`: Contains the `Group` class, which represents a mathematical group and provides various methods for group operations.
    - `permutation.py`: Contains the `Permutation` class, which represents permutations and provides methods for permutation operations.
    - `symbolic.py`: Contains the `Term` and `Expression` classes, which represent symbolic terms and expressions used in group theory.
    - `utils.py`: Utility functions used throughout the project, including functions for factorization, generating subgroups, and more.
- `playground.ipynb`: Jupyter notebook for experimenting with the group theory code and running tests.
- `tests/`
    - `generate_tests.py`: Automatic code generation for making some good multiplication tests.
    - `test_multiply.py`: Tests for multiplication and parsing, for both permutation and symbolic groups.

## Installation

To use this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/JacksonKaunismaa/group_theory
cd group_theory
pip install -r requirements.txt
```

## Usage
### Creating Groups

You can create different types of groups using the `get_group` function from `utils.py`. For example:
```python
from group_theory.utils import get_group

# Create a dihedral group of order 8
d8 = get_group("dih 8")

# Create a cyclic group of order 6
c6 = get_group("cyc 6")

# Custom symbolic group
from group_theory.symbolic import SymbolicGroup
group = SymbolicGroup(rules=["a8 = e", "b3 = e", "b a = a7 b2"], name='custom 8')
```

When creating custom groups, you must be careful to specify rules that don't result
in infinite groups or contradictions. `SymbolicGroup` does not automatically
detect this, and if you attempt to generate all elements of such a group, it will
run indefinitely.


### Group Operations

The `Group` class provides various methods for group operations, such as generating subgroups, finding cosets, and checking normality:

```python
# Generate all elements of the group
d8._generate_all()

# Find the centralizer of an element
centralizer = d8.centralizer("r2")

# Check if a subgroup is normal
is_normal = d8.is_normal(subgroup)
```

### Symbolic Expressions

The `Term` and `Expression` classes represent symbolic terms and expressions, but should not be typically accessed. The preferred way to create symbolic expressions is through `Group.evaluate(equation: str)`
```python
from group_theory.symbolic import Term, Expression

# Create a term manually
term = Term("r", 3, group=d8)

# Create an expression manually
m_expr = Expression("r3 f", group=d8)

# Simplify the expression
simplified_expr = m_expr.simplify()

# Preferred way to create expressions
expr = d8.evaluate('r3 f')
```

### Permutations

There are three ways to manually create `Permutation`, result notation, cycle notation, or through string parsing. However, the preferred way to create `Permutation` is through `Group.evaluate(equation: str)`, where `Group` corresponds to a permutation group.
```python
from group_theory.permutation import  Permutation

# Create a permutation manually, through result notation
# result notation is a list that indicates where each element ends up after the permutation
p_expr = Permutation([0, 1, 2, 7, 4, 5, 6, 3], get_group('s8'))
p_expr = p_expr.simplify()  # (4 8)

# Create a permutation manually, through cycle notation
p_expr = Permutation([[1, 2, 3]], get_group('s8'))  # (2 3 4)

# Create a permutation manually, through a string
p_expr = Permutation("(2 3 4 5)", get_group('s8'))  # (2 3 4 5)

# Preferred way to create Permutations
s8 = get_group('s 8')
p_expr = s8.evaluate('(2 6 1)')  # (1 2 6)
```

### Manipulating expressions

You can multiply, divide, and find inverses for both `Permutation` and `Expression`.
```python
x = d8.evaluate('r3 f')
y = d8.evaluate('r2 f')
z = d8.evaluate('r6')

x * y  # r
z.inv()  # r2
x / y # equivalent to x * y.inv(), r
((x * y) / z)  # r3
```

The library attempts to be as permissive as possible as to what types of
operations are allowed, promoting strings and other types that could possibly be
parsed as `Permutation` or `Expression` to the appropriate type. This allows stuff like
this:
```python
x = d8.evaluate('r3 f')
x * 'f' / 'r'  # r2 as an Expression

y = s8.evaluate("(1 2 3)")
[[2,3]] * y / '(2 3)'   # (1 3 4) as a Permutation, using cycle notation and string notation
[0, 1, 3, 4, 2, 5, 7, 6] * y / '(2 3)'  # (1 3 5 4)(7 8), using result notation and string notation
```

Result notation is an alternative way of expressing permutations by indicating where each
element ends up after the permutation. For example, the permutation `(1 2)` can be expressed
as `[1, 0, 2, 3]` in result notation for the group s4. Note that result notation is 0-indexed, whereas
cycle notation is 1-indexed.

### Running Tests

You can run the tests via `pytest`, running from the root directory:
```bash
pytest
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.