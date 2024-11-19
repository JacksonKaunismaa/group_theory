import pytest
from group_theory.group_utils import get_group


@pytest.mark.parametrize(
    "test_group, elems, is_subgroup",
    [
        ("s 5", ["(1 2)", "(1 2 3)", "(1 5)"], False),
        ("d 12", ["r3 f", "e"], True),
        ("s 3", ["(1 2)", "(1 3)", "(1 2 3)", "(2 3)", "(1 3 2)"], False),
        ("s 3", ["(1 2)", "(1 3)", "(1 2 3)", "(2 3)", "(1 3 2)", "()"], True),
        ("d 4", ["r", "r2"], False),
    ],
)
def test_group_subgroup(test_group, elems, is_subgroup):
    gr = get_group(test_group, generate=True)
    subgroup = gr.subgroup(*elems)
    assert subgroup.is_subgroup() == is_subgroup


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


@pytest.mark.parametrize(
    "test_group, subgroup_elems, is_normal",
    [
        ("s 3", ["(1 2)"], False),
        ("s 3", ["(1 2)", "(1 3)"], True),
        ("d 12", ["r3", "f"], False),
        ("d 12", ["r3"], True),
        ("d 12", ["r", "f"], True),
    ],
)
def test_group_is_normal(test_group, subgroup_elems, is_normal):
    gr = get_group(test_group, generate=True)
    subgroup = gr.generate(*subgroup_elems)
    assert gr.is_normal(subgroup) == is_normal


@pytest.mark.parametrize(
    "test_group, elems",
    [
        ("s 5", ["(1 2)", "(2 4)"]),
        ("s 5", ["(1 2 3)", "(1 5)"]),
        ("sd 4", ["r", "s", "r7 s"]),
        ("dic 8", ["s", "s r"]),
    ],
)
def test_group_centralizer(test_group, elems):
    gr = get_group(test_group, generate=True)
    centralizer = gr.centralizer(elems)

    # check that centralizer < C_G(elems)
    for g in centralizer:
        for x in elems:
            assert g * x == x * g

    # check that C_G(elems) < centralizer
    for g in gr:
        if all(g * x == x * g for x in elems):
            assert g in centralizer


@pytest.mark.parametrize(
    "test_group, base_elem, orbit_elems",
    [
        ("s 5", "(1 2 3)", ["(1 2 3)", "(1 3 2)", "()"]),
        ("sd 4", "r", ["r", "r2", "r3", "e"]),
        ("dic 8", "r", ["r", "r2", "r3", "r4", "r5", "r6", "r7", "e"]),
    ],
)
def test_group_orbit(test_group, base_elem, orbit_elems):
    gr = get_group(test_group)
    orbit = gr.orbit(base_elem)
    assert orbit == gr.subgroup(*orbit_elems)
