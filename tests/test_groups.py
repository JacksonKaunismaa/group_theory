import pytest
from group_theory.group_utils import get_group


@pytest.mark.parametrize(
    "test_group,elems",
    [
        ("s 5", ["(1 2)", "(1 2 3)", "(1 5)"]),
        ("d 12", ["r3", "f"]),
    ],
)
def test_group_is_subgroup(test_group, elems):
    gr = get_group(test_group, generate=True)
    subgroup = gr.generate(*elems)
    assert subgroup.is_subgroup()


@pytest.mark.parametrize(
    "test_group, elems, tests",
    [
        ("s 3", ["(1 2)"], [("(1 2)", True), ("(1 3)", False), ("(1 2 3)", False)]),
        ("s 3", ["(1 2)", "(1 3)"], [("(1 2)", True), ("(1 2 3)", True)]),
        ("d 4", ["r"], [("r", True), ("r2", True), ("f", False)]),
        ("d 4", ["r", "f"], [("r", True), ("f", True), ("r f", True), ("r2", True)]),
    ],
)
def test_group_generate(test_group, elems, tests):
    gr = get_group(test_group)
    subgroup = gr.generate(*elems)
    for elem, is_in in tests:
        assert (elem in subgroup) == is_in
