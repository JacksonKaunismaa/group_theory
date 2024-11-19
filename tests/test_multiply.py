import pytest
from group_theory.group_utils import get_group
from group_theory.symbolic import SymbolicGroup


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("e e e e f e e e e e e e f e e e", "e"),
        ("r25 f1 r2 r4 r-17 f1 f2 f12", "r4"),
        ("e", "e"),
        ("1", "e"),
        ("f2 r8 e", "e"),
        ("r1", "r"),
        ("r2 f", "r2 f"),
        ("r3 f e r2 f e f", "r f"),
        ("r9 f r r r r", "r5 f"),
        ("r r r2 f r3 f f r5 r4 r", "r7 f"),
    ],
)
def test_parsing(test_input, expected):
    d8_alt = SymbolicGroup(rules=["r8 = e", "f2 = e", "r f r = f", "f r = r7 f"])
    d8 = get_group("d8")

    parsed_alt = d8_alt.parse(test_input)
    parsed = d8.parse(test_input)
    assert parsed == parsed_alt

    answer_alt = d8_alt.parse(expected)
    answer = d8.parse(expected)

    evald = d8_alt.evaluate(test_input)
    evald_alt = d8.evaluate(test_input)

    simplified_alt = parsed_alt.simplify()
    simplified = parsed.simplify()

    assert answer == simplified == answer_alt == simplified_alt == evald == evald_alt
    assert (
        str(answer)
        == str(simplified)
        == str(answer_alt)
        == str(simplified_alt)
        == str(evald)
        == str(evald_alt)
        == str(expected)
    )


@pytest.mark.parametrize(
    "group_name,tests",
    [
        (
            "dic 28",
            [
                ("r11 s", "e", "r11 s"),
                ("r10 s", "r5 s", "r19"),
                ("r9 s", "r17 s", "r6"),
                ("r11 s", "r12", "r27 s"),
                ("r13", "r23 s", "r8 s"),
            ],
        ),
        (
            "sa 32",
            [
                ("r10", "r11", "r21"),
                ("r19 s", "r11", "r14 s"),
                ("r24 s", "r2", "r26 s"),
                ("r16 s", "r16", "s"),
                ("r12 s", "r5 s", "r"),
            ],
        ),
        (
            "sd 64",
            [
                ("r17 s", "r35", "r14 s"),
                ("r29", "r3", "r32"),
                ("r6 s", "r36 s", "r34"),
                ("r46", "r53 s", "r35 s"),
                ("r32", "r28 s", "r60 s"),
            ],
        ),
        (
            "dih 31",
            [
                ("r24 f", "r28 f", "r27"),
                ("r9 f", "r5 f", "r4"),
                ("r14", "r16", "r30"),
                ("r3", "r25", "r28"),
                ("r24 f", "r17", "r7 f"),
            ],
        ),
        (
            "cyc 29",
            [
                ("r15", "r2", "r17"),
                ("r11", "r7", "r18"),
                ("r8", "r4", "r12"),
                ("r2", "r4", "r6"),
                ("r12", "r4", "r16"),
            ],
        ),
        (
            "abelian 14",
            [
                ("r4", "s", "r4 s"),
                ("r9", "r13 s", "r8 s"),
                ("r s", "r9", "r10 s"),
                ("e", "r", "r"),
                ("r11 s", "r12 s", "r9"),
            ],
        ),
        (
            "quat 32",
            [
                ("r20", "r18 s", "r6 s"),
                ("r4 s", "r5", "r31 s"),
                ("r14", "r31", "r13"),
                ("r5 s", "r29 s", "r24"),
                ("r22", "r30 s", "r20 s"),
            ],
        ),
    ],
)
def test_symbolic_multiplication(group_name, tests):
    gr = get_group(group_name)
    for t1, t2, ans in tests:
        t1, t2, ans = gr.evaluates(t1, t2, ans)
        assert t1 * t2 == ans


@pytest.mark.parametrize(
    "group_name,tests",
    [
        (
            "sym 8",
            [
                ("(1 8 5 4 3 7 2)(6)()", "(1 6 4 7 5 3)(2)(8)()", "(1 8 3 5 7 2 6 4)"),
                (
                    "(1 2 6 8 3 5 7)(4)()",
                    "(1)(2)(3 4 8 7 6)(5)()",
                    "(1 2 3 5 6 7)(4 8)",
                ),
                ("(1 7 5 6)(2 8 3 4)", "(1 8 6 4)(2)(3 5)(7)()", "(1 7 3)(2 6 8 5 4)"),
                ("(1 3 5 8)(2 6 4 7)", "(1 2)(3 7 8 4 6 5)", "(1 7)(2 5 4 8)"),
                ("(1 6)(2 3 8 4 7)(5)()", "(1 6 7 4 3 5)(2)(8)()", "(1 7 2 5)(3 8)"),
            ],
        ),
        (
            "alt 8",
            [
                ("(1 2 8 4)(3)(5)(6 7)", "(1)(2 8 6 7 5 3 4)", "(1 8 2 6 5 3 4)"),
                ("(1 3 7)(2 4 8)(5)(6)()", "(1 5 8 7 4 3)(2 6)", "(2 3 4 7 5 8 6)"),
                ("(1 3 5 4 2 7 8 6)", "(1 2 8 5 4)(3)(6)(7)()", "(1 3 4 8 6 2 7 5)"),
                ("(1 6 8 5 3 4 7)(2)()", "(1 5 4 3 8 7)(2)(6)()", "(1 6 7 5 8 4)"),
                ("(1 8 3 7 2)(4)(5)(6)()", "(1 3 4 6 8 2)(5 7)", "(1 2 3 5 7)(4 6 8)"),
            ],
        ),
    ],
)
def test_permutation_multiplication(group_name, tests):
    gr = get_group(group_name)
    for t1, t2, ans in tests:
        t1, t2, ans = gr.evaluates(t1, t2, ans)
        assert t1 * t2 == ans


@pytest.mark.parametrize(
    "group_name,tests",
    [
        (
            "sym 5",
            [
                ([1, 3, 4, 0, 2], "(1 4 2)(3 5)"),
                ([2, 4, 0, 1, 3], "(1 3)(2 4 5)"),
                ([4, 2, 3, 1, 0], "(1 5)(2 4 3)"),
            ],
        ),
        (
            "sym 6",
            [
                ([5, 1, 0, 2, 3, 4], "(1 3 4 5 6)"),
                ([3, 5, 4, 2, 0, 1], "(1 5 3 4)(2 6)"),
                ([1, 5, 4, 0, 3, 2], "(1 4 5 3 6 2)"),
            ],
        ),
        (
            "sym 7",
            [
                ([5, 6, 2, 0, 1, 3, 4], "(1 4 6)(2 5 7)"),
                ([4, 2, 3, 5, 6, 0, 1], "(1 6 4 3 2 7 5)"),
                ([4, 6, 3, 5, 1, 0, 2], "(1 6 4 3 7 2 5)"),
            ],
        ),
    ],
)
def test_permutation_result_notation_parsing(group_name, tests):
    gr = get_group(group_name)
    for result_notation, cycle_notation in tests:
        perm = gr.evaluate(result_notation).simplify()
        assert str(perm) == cycle_notation


def test_symbolic_permissive_str_multiply():
    gr = get_group("dic 8")
    t1, t2, ans = ("r3 s", "r", "r2 s")
    t1 = gr.evaluate(t1)  # leave t2 and ans as strings
    assert t1 * t2 == ans


def test_permutation_permissive_str_multiply():
    gr = get_group("sym 8")
    t1 = gr.evaluate("(1 2 8 4)(3)(5)(6 7)")
    t2 = "(1)(2 8 6 7 5 3 4)"
    ans = "(1 8 2 6 5 3 4)"
    assert t1 * t2 == ans


def test_permutation_permissive_cycle_multiply():
    gr = get_group("sym 8")
    t1 = gr.evaluate("(1 2 8 4)(3)(5)(6 7)")
    t2 = [[0], [1, 7, 5, 6, 4, 2, 3]]
    ans = [[0, 7, 1, 5, 4, 2, 3]]
    assert t1 * t2 == ans


def test_permutation_permissive_result_multiply():
    gr = get_group("sym 8")
    t1 = gr.evaluate("(1 2 8 4)(3)(5)(6 7)")
    t2 = [[1, 2, 0, 6, 7, 5, 3, 4]]
    ans = "(1 3)(2 6 8 5)(4 7)"
    assert t1 * t2 == ans
