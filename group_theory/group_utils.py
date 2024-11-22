import re
import math
from typing import TYPE_CHECKING, Optional

from group_theory import permutation
from group_theory import symbolic

if TYPE_CHECKING:
    from group_theory.groups import Group


def _default_groups(group_name: str, n: int, generate: Optional[bool]) -> "Group":
    # some definitions for common finite groups
    if group_name == "quaternion":
        group_name = "dicyclic"
    # symbolic groups definition
    rules_d = dict(
        cyclic=[f"r{n} = e"],
        dihedral=[f"r{n} = e", "f2 = e", f"f r = r{n-1} f"],
        dicyclic=[f"r{n} = e", "s4 = e", f"s2 = r{n//2}", f"s r = r{n-1} s"],
        semi_dihedral=[f"r{n} = e", "s2 = e", f"s r = r{n//2-1} s"],
        semi_abelian=[f"r{n} = e", "s2 = e", f"s r = r{n//2+1} s"],
        abelian=[f"r{n} = e", "s2 = e", "s r = r s"],
    )

    if group_name in ["symmetric", "alternating"]:
        if generate is None:
            generate = False
        return permutation.PermutationGroup(
            n=n, generate=generate, name=f"{group_name} {n}"
        )
    else:
        if group_name in ["dicyclic"] and n % 2 != 0:
            raise ValueError(
                f"n must be divisible by 2 for {group_name}"
                f"group, it was {n} instead"
            )
        if group_name in ["semi_dihedral", "semi_abelian"] and math.log2(n) != int(
            math.log2(n)
        ):

            raise ValueError(
                f"n must be a power of 2 for {group_name}" f"group, it was {n} instead"
            )

        if generate is None:
            generate = True

        return symbolic.SymbolicGroup(
            rules=rules_d[group_name], generate=generate, name=f"{group_name} {n}"
        )


def get_group(query: str, generate=None) -> "Group":
    """
    Get a group object from a query string.
    Examples:
        get_group("d6")
        get_group("dihedral 6")
        get_group("s4")

    Args:
        query (str): The query string.
        generate (bool): Whether to generate the group elements.
    Returns:
        Group: The group object.
    """
    extracted = re.search(r"([a-zA-Z]+) ?(\d+)", query.lower().strip())
    mappings = [
        (["d", "dihedral", "dih", "di"], "dihedral"),
        (["c", "z", "cyclic", "cyc"], "cyclic"),
        (["dic", "dicyclic"], "dicyclic"),
        (["semi-dihedral", "sd", "semi_dihedral"], "semi_dihedral"),
        (["semi-abelian", "sa", "semi_abelian"], "semi_abelian"),
        (["abelian", "ab"], "abelian"),
        (["quaternion", "quat", "q"], "quaternion"),
        (["perm", "permutation", "sym", "symmetric", "s"], "symmetric"),
        (["alt", "alternating", "a"], "alternating"),
    ]
    if not extracted:
        raise ValueError("Group name not recognized")
    group_name = extracted.group(1).strip()
    for alt_names, name in mappings:
        if group_name in alt_names:
            group_name = name
            break
    else:
        raise ValueError(f"Group name {group_name} not recognized")

    size = int(extracted.group(2))
    return _default_groups(group_name, size, generate)
