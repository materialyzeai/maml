"""Module implements helper functions to retrieve data for GB energy prediction paper."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from monty.serialization import dumpfn, loadfn
from pymatgen.ext.matproj import MPRester, MPRestError

from maml.apps import gbe

if TYPE_CHECKING:
    from pymatgen.core import Structure

pjoin = os.path.join
module_dir = os.path.dirname(gbe.__file__)
REFS = pjoin(module_dir, "references")
DATA = pjoin(module_dir, "data")


def load_data(filename: str | None = None) -> list:
    """
    Helper function to load the data
    Default is to load the 361 data
    Args:
        filename (str): the filename of the data.

    Returns:
        data list

    """
    if filename:
        return loadfn(filename)
    return loadfn(pjoin(DATA, "gb_data.json"))


def load_b0_dict() -> dict:
    """
    Helper function to retrieve the b0 data.
    b0 is the bulk bond length.

    Returns:
        returns the dict of {element: b0}
    """
    if os.path.isfile(pjoin(REFS, "el2b0.json")):
        return loadfn(pjoin(REFS, "el2b0.json"))
    print("Downloading Data ... ")
    if os.getenv("MAPI_KEY"):
        data = load_data()
        b0_dict = update_b0_dict(data, mp_api=os.getenv("MAPI_KEY"))
        dumpfn(b0_dict, pjoin(REFS, "el2b0.json"))
        return b0_dict
    raise MPRestError("No MAPI_KEY found")


def update_b0_dict(data_source: list, mp_api: str | None) -> dict:
    """
    Helper function to update the b0 dictionary
    Requires api key to MP
    Args:
        data_source (list): list of GB data
        mp_api (str): API key to MP
    Returns:
         b0_dict (dict): {el: b0}.
    """
    rester = MPRester(mp_api)

    def get_b0(bulk_structure: Structure) -> float:
        """b0 is the bond length of bulk metal."""
        _b0 = bulk_structure.get_distance(0, 0, jimage=1)
        return min(_b0, bulk_structure.lattice.a)

    b0_dict = {}
    for d in data_source:
        if d["pretty_formula"] in b0_dict:
            continue
        bs = rester.get_structure_by_material_id(d["material_id"])  # type: ignore[attr-defined]
        b0 = get_b0(bs)
        b0_dict.update({d["pretty_formula"]: b0})
    return b0_dict


def load_mean_delta_bl_dict(loc_algo: str = "crystalnn") -> dict:
    """
    Helper function to load the mean_delta_bl data
    Args:
        loc_algo (str): name of the algorithm.

    Returns:
        {task_id (int): mean_delta_bl(float)}

    """
    if os.path.isfile(pjoin(REFS, f"mean_bl_chg_{loc_algo}.json")):
        return loadfn(pjoin(REFS, f"mean_bl_chg_{loc_algo}.json"))
    raise ValueError(
        "Please provide mean_delta_bond_length data. "
        "Use gbe.describers.GBond.get_mean_bl_chg method"
        "to calculate mean_delta_bond_length"
    )
