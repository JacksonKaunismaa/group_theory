from group_theory import groups, permutation, utils

# Test code generators

def rand_expr(group: groups.Group):
    import random
    if group.is_perm_group:
        res = list(range(group.n))
        random.shuffle(res)
        return permutation.Permutation(res, group)
    else:
        expr_str = ""
        for sym in group.symbols:
            expr_str += f"{sym}{random.randint(0,100)} "
        return group.evaluate(expr_str)


def rand_problem(group: groups.Group):
    t1 = rand_expr(group)
    t2 = rand_expr(group)
    return (str(t1), str(t2), str(t1*t2))


def multiplication_problem_template(group_name, problems):
    problem_text = [f"t1,t2,ans = gr.evaluate({problem}); assert t1*t2 == ans" for problem in problems]
    return "\n".join([f"gr = get_group('{group_name}')"] + problem_text)


def generate_problems(num_problems=2):  # code generator for tests
    group_names = ["dic 28", "sa 32", "sd 64", "dih 31", "cyc 29", "abelian 14", "quat 32", "sym 8", "alt 8"]
    for group_name in group_names:
        print(group_name)
        utils.get_group(group_name)
    group_selection = list(map(utils.get_group, group_names))
    probs = [[rand_problem(gr) for _ in range(num_problems)] for gr in group_selection]
    print("# Test multiplication (generated code)")
    for group_name,prob in zip(group_names, probs):
        print(multiplication_problem_template(group_name, prob))
        print()
