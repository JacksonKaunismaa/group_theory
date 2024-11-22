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


@pytest.mark.parametrize(
    "test_group, subgroup_gen, mult, coset_elems",
    [
        (
            "s 3",
            ["(1 2)"],
            "(1 2 3)",
            [
                ("(1 2 3)", (True, True)),
                ("(1 3 2)", (False, False)),
                ("(1 3)", (True, False)),
            ],
        ),
        (
            "d 4",
            ["r"],
            "f",
            [("f", (True, True)), ("r f", (True, True)), ("r2", (False, False))],
        ),
        (
            "d 8",
            ["r2", "f"],
            "r",
            [("r", (True, True)), ("r f", (True, True)), ("f", (False, False))],
        ),
    ],
)
def test_cosets_multiply(test_group, subgroup_gen, mult, coset_elems):
    gr = get_group(test_group)
    subgroup = gr.generate(*subgroup_gen)
    right_coset = subgroup * mult
    left_coset = mult * subgroup
    for elem, (is_in_right, is_in_left) in coset_elems:
        assert (elem in right_coset) == is_in_right
        assert (elem in left_coset) == is_in_left


@pytest.mark.parametrize(
    "test_group, subgroup_gen, div, coset_elems",
    [
        (
            "s 3",
            ["(1 2)"],
            "(1 2 3)",
            [
                ("(1 2 3)", (False, True)),
                ("(1 3 2)", (True, False)),
                ("(1 3)", (False, False)),
                ("(2 3)", (True, True)),
            ],
        ),
        (
            "d 4",
            ["r"],
            "f",
            [("f", (True, True)), ("r f", (True, True)), ("r2", (False, False))],
        ),
        (
            "d 8",
            ["r2", "f"],
            "r",
            [("r", (True, True)), ("r f", (True, True)), ("f", (False, False))],
        ),
    ],
)
def test_cosets_divide(test_group, subgroup_gen, div, coset_elems):
    gr = get_group(test_group)
    subgroup = gr.generate(*subgroup_gen)
    right_coset = subgroup / div
    left_coset = div / subgroup
    for elem, (is_in_right, is_in_left) in coset_elems:
        assert (elem in right_coset) == is_in_right
        assert (elem in left_coset) == is_in_left


@pytest.mark.parametrize(
    "test_group, center_elems",
    [
        ("s 3", ["e"]),
        ("d 4", ["e", "r2"]),
        ("d 8", ["e", "r4"]),
        ("dic 8", ["e", "r4"]),
        ("quat 8", ["e", "r4"]),
    ],
)
def test_center(test_group, center_elems):
    gr = get_group(test_group, generate=True)
    center = gr.center()
    expected_center = gr.generate(*center_elems)
    assert center == expected_center


@pytest.mark.parametrize(
    "test_group, elems, normal_core_elems",
    [
        ("s 3", ["(1 2)"], ["e"]),
        ("d 12", ["r3", "f"], ["r3"]),
        ("dic 12", ["r4", "s"], ["r2"]),
        ("dic 12", ["r3", "s"], ["r3"]),
    ],
)
def test_normal_core(test_group, elems, normal_core_elems):
    gr = get_group(test_group, generate=True)
    generated = gr.generate(*elems)
    normal_core = gr.normal_core(generated)
    expected_normal_core = gr.generate(*normal_core_elems)
    assert normal_core == expected_normal_core
    assert gr.is_normal(expected_normal_core)
    assert gr.is_normal(normal_core)
    assert normal_core <= generated


@pytest.mark.parametrize(
    "test_group, elems, normal_closure_elems",
    [
        ("s 3", ["(1 2)"], ["(1 2)", "(2 3)"]),
        ("d 12", ["r4", "f"], ["r2", "f"]),
        ("quat 16", ["r12", "s"], ["r2", "s"]),
    ],
)
def test_normal_closure(test_group, elems, normal_closure_elems):
    gr = get_group(test_group, generate=True)
    generated = gr.generate(*elems)
    normal_closure = gr.normal_closure(generated)
    expected_normal_closure = gr.generate(*normal_closure_elems)
    assert normal_closure == expected_normal_closure
    assert gr.is_normal(normal_closure)
    assert gr.is_normal(expected_normal_closure)
    assert generated <= normal_closure


@pytest.mark.parametrize(
    "test_group, elems, normalizer_elems",
    [
        ("s 4", ["(1 2)", "(3 4)"], ["(1 2)", "(1 3 2 4)"]),
        ("d 8", ["r4", "f"], ["r2", "f"]),
        ("dic 12", ["r2 s"], ["r2 s", "r3"]),
    ],
)
def test_normalizer(test_group, elems, normalizer_elems):
    gr = get_group(test_group, generate=True)
    generated = gr.generate(*elems)
    normalizer = gr.normalizer(generated)
    expected_normalizer = gr.generate(*normalizer_elems)
    assert normalizer.is_normal(generated)
    assert normalizer == expected_normalizer


@pytest.mark.parametrize(
    "test_group, commutator_core",
    [
        ("s 4", ["(1 3 2)", "(1 2)(3 4)"]),
        ("d 12", ["r2"]),
        ("sa 16", ["r8"]),
    ],
)
def test_commutator(test_group, commutator_core):
    gr = get_group(test_group, generate=True)
    commutator = gr.commutator()
    expected_commutator = gr.generate(*commutator_core)
    assert commutator == expected_commutator
    expected_closure = gr.normal_closure(expected_commutator)
    assert commutator == expected_closure
