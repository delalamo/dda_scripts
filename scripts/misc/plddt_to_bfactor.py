# Author: Diego del Alamo - diego.delalamo@gmail.com
# License: None

# This script runs principal component analysis on a set of
# 	protein models after structural alignment by TMAlign

import argparse
import Bio.PDB
import numpy as np
import os

from absl import logging
from sklearn.decomposition import PCA
from typing import Callable, Dict, List, NoReturn, Tuple


def plddt_to_b(plddt: float, maxval: float = 100.0) -> float:
    r"""Converts a pLDDT vals to a B factor

    Parameters
    ----------
    plddt : pLDDT value to convert
    maxval : Set to 100 if using AF2 (or 1 if RoseTTAFold)

    Returns
    ----------
    B-factor

    """

    rmsf = 1.5 * np.exp(4 * (0.7 - (plddt / maxval)))
    return (8.0 / 3.0) * (np.pi**2) * (rmsf**2)


def _main() -> NoReturn:
    r"""Converts the LDDT vals of an AF2 PDB to B-factors

    Parameters
    ----------
    None

    Returns
    ----------
    None

    """

    parser = argparse.ArgumentParser(
        prog="af2_to_bfactor.py",
        description=(
            "This program converts the pLDDT values of a PDB "
            "into B-factors. Uses the equation found in "
            "https://doi.org/10.1038/s41467-021-21511-x"
        ),
        epilog="",
    )

    parser.add_argument(
        "-i",
        "--input_pdb",
        metavar="model1.pdb",
        type=str,
        help="(required) PDB file for conversion",
        required=True,
        dest="filename",
    )

    args = parser.parse_args()

    pdb = Bio.PDB.PDBParser().get_structure("TEMP", args.filename)
    for atom in pdb.get_atoms():
        atom.bfactor = plddt_to_b(atom.bfactor)

    pdbio = Bio.PDB.PDBIO()
    pdbio.set_structure(pdb)
    pdbio.save(args.filename)


if __name__ == "__main__":
    _main()
