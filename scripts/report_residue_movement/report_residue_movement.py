# Author: Diego del Alamo - diego.delalamo@gmail.com
# License: None

# This script reports per-residue movement between two structures.

import argparse
import Bio.PDB
import numpy as np
import os
import pandas as pd

from absl import logging

from typing import Callable, Dict, List, NoReturn, Tuple

"""
leut - 2a65/3tt3
betp - 4ain/4llh
lat1 - 6irs/7dsq
dranramp - 6d9w/6d9i
sert - 5i6x/6dzz
mhp1 - 2jln/2x79
"""

# Ask for file and chain of struct 1
# Ask for file and chain of struct 2
# Use Bio PDB to isolate the chains and use TMAlign align the two
# Iterate over residues that match
# Print net movement of residues

# Author: Diego del Alamo - diego.delalamo@gmail.com
# License: None

# This script threads a protein sequence onto a structure using
# TM-Align and Rosetta.


def _print_and_run(fn: Callable[[str], None], cmd: str) -> NoReturn:
    r"""Small function for printing and calling a command

    Parameters
    ----------
    cmd : Command to execute
    fn : Print function

    Returns
    ----------
    None

    """

    fn(cmd)
    os.system(cmd)


def _args() -> Tuple[str, str, str, str]:
    r"""Process input arguments for alignment

    Parameters
    ----------
    None

    Returns
    ----------
    vals : Dictionary containing organized command-line options

    """

    parser = argparse.ArgumentParser(
        prog="report_residue_movement.py",
        description=(
            "This program calculates and reports per-residue "
            "movements in angstroms between two structures. "
            "Structures are aligned using TM-Align before "
            "calculating."
        ),
        epilog="",
    )

    parser.add_argument(
        "-x",
        "--pdb1",
        metavar="example.pdb",
        type=str,
        help="(required) First PDB file",
        required=True,
        dest="pdb1",
    )

    parser.add_argument(
        "-y",
        "--pdb2",
        metavar="example.pdb",
        type=str,
        help="(required) Second PDB file",
        required=True,
        dest="pdb2",
    )

    parser.add_argument(
        "-t",
        "--tmalign",
        metavar="/path/to/tmalign/TMalign",
        type=str,
        help="(required) TM-Align executable",
        required=True,
        dest="tmalign",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="(optional) Set verbosity",
        action="store_true",
        dest="verbose",
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="out.csv",
        type=str,
        help="(optional) Name of output CSV file",
        required=False,
        default="out.csv",
        dest="outname",
    )

    args = parser.parse_args()

    args = {arg: getattr(args, arg) for arg in vars(args)}

    if args["verbose"]:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)
    del args["verbose"]

    return args


def _tmalign(tmalign_exe: str, pdb1: str, pdb2: str) -> str:
    r"""Aligns one structure to another

    Parameters
    ----------
    tmalign_exe : Executable for alignment
    pdb1 : First PDB file
    pdb2 : Second PDB file

    Returns
    ----------
    outname : Name of output file
    """

    outname = "temp_{}".format(os.path.basename(pdb1))
    cmd = " ".join((tmalign_exe, pdb1, pdb2, "-o", outname))
    _print_and_run(logging.debug, cmd)
    return outname


def _res_xyz(pdb: str) -> Dict[int, np.array]:
    r"""Parses both models and computes movement between alpha carbons

    Parameters
    ----------
    pdb : PDB file

    Returns
    ----------
    res_xyz : Dictionary mapping residues to alpha carbon coords

    """

    res_xyz = dict()

    pdbparser = Bio.PDB.PDBParser()
    model = pdbparser.get_structure("TEMP", pdb)

    for residue in model.get_residues():
        res = residue.get_id()[1]
        try:
            xyz = [residue["CA"].coord[i] for i in range(3)]
            res_xyz[res] = np.asarray(xyz)
        except:
            continue
    return res_xyz


def _calc_changes(pdb1: str, pdb2: str) -> Dict[int, float]:
    r"""Parses both models and computes movement between alpha carbons

    Parameters
    ----------
    pdb1 : First PDB file
    pdb2 : Second PDB file

    Returns
    ----------
    res_movement_dict : Dictionary mapping residues to alpha carbon movement

    """

    res_movement_dict = dict()

    xyz1 = _res_xyz(pdb1)
    xyz2 = _res_xyz(pdb2)

    for res, coord1 in xyz1.items():
        if res not in xyz2:
            continue
        coord2 = xyz2[res]
        res_movement_dict[res] = np.linalg.norm(coord1 - coord2)
    return res_movement_dict


def _main() -> NoReturn:
    r"""Main function for reading and writing outputs.
    During execution, several temporary files will be written
    in the current working directory. These will then be deleted
    during cleanup.

    Parameters
    ----------
    None

    Returns
    ----------
    None

    """

    # Setting temporary names
    c1, c2 = "Residue", "Movement (CA)"
    df = pd.DataFrame(columns=[c1, c2])

    args = _args()
    outname = _tmalign(args["tmalign"], args["pdb1"], args["pdb2"])

    for res, xyz in _calc_changes(outname, args["pdb2"]).items():
        df = df.append({c1: res, c2: xyz}, ignore_index=True)

    # Clean up and save
    df["Residue"] = df["Residue"].astype(int)
    df.set_index("Residue", drop=True).to_csv(args["outname"])

    # Remove temporary variable
    os.system(f"rm { outname }")


if __name__ == "__main__":
    _main()
