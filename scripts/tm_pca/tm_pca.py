# Author: Diego del Alamo - diego.delalamo@gmail.com
# License: None

# This script threads a protein sequence onto a structure using
# TM-Align and Rosetta. 

import argparse
import Bio.PDB
import numpy as np
import os

from absl import logging
from sklearn.decomposition import PCA
from typing import Callable, Dict, List, NoReturn, Tuple

def _print_and_run(
		fn : Callable[ [ str ], None ],
		cmd : str
	) -> NoReturn:
	r""" Small function for printing and calling a command

	Parameters
	----------
	cmd : Command to execute
	fn : Print function

	Returns
	----------
	None

	"""

	fn( cmd )
	os.system( cmd )

def _args() -> Tuple[ List[ str ], str, List[ int ], str, str ]:
	r""" Process input arguments for alignment

	Parameters
	----------
	None

	Returns
	----------
	vals : Dictionary containing organized command-line options
	
	"""

	parser = argparse.ArgumentParser(
			prog="tmalign_pca.py",
			description= (
					"This program performs a structural alignment on a "
					"bunch of models and runs principal component "
					"analysis on the structures."
				),
			epilog=""
		)

	parser.add_argument(
			'-i',
			'--input_pdbs',
			metavar="model1.pdb model2.pdb ... modelN.pdb",
			type=str,
			nargs="+",
			help="(required) List of PDBs for alignment",
			required=True,
			dest="models"
		)

	parser.add_argument(
			'-t',
			'--tmalign',
			metavar="/path/to/tmalign/TMalign",
			type=str,
			help="(required) TM-Align executable",
			required=True,
			dest="tmalign"
		)

	parser.add_argument(
			'-p',
			'--pdb',
			metavar="reference.pdb",
			type=str,
			help=( "(optional) Reference PDB for structural alignment. "
					"If left blank, the first model will be used as a "
					"reference for the other models (not recommended)."
				),
			dest="ref"			
		)

	parser.add_argument(
			'-r',
			'--res_to_keep',
			metavar="1 2 3 ... N",
			type=int,
			nargs="+",
			help=( "(optional) Residues to keep during PCA. "
					"Default: all residues are used."
				),
			required=False,
			dest="keep"
		)

	parser.add_argument(
			'--res_to_ignore',
			metavar="1 2 3 ... N",
			type=int,
			nargs="+",
			help=( "(optional) Residues to ignore during PCA. "
					"Note that this flag and --res_to_keep are mutually "
					"exclusive; using both causes this option to be ignored."
				),
			required=False,
			dest="ignore"
		)

	parser.add_argument(
			'-v',
			'--verbose',
			help="(optional) Set verbosity",
			action="store_true",
			dest="verbose"
		)

	args = parser.parse_args()

	args = { arg: getattr( args, arg ) for arg in vars( args ) }

	# Now go through and clean things up
	if args[ "verbose" ]:
		logging.set_verbosity( logging.DEBUG )
	else:
		logging.set_verbosity( logging.INFO )

	if "ref" not in args:
		args[ "ref" ] = args[ "models" ].pop( 0 )
		logging.warning( "No reference PDB provided! Using model {}.".format(
				args[ "ref" ]
			) )

	if "res_to_ignore" in args:
		if "res_to_keep" in args:
			logging.warning( "Both --res_to_keep and --res_to_ignore options "
					"used! Ignoring --res_to_ignore."
				)
		else:
			ref_r = list( read_pdb( args[ "ref" ] ).keys() )
			args[ "keep" ] = [ r for r in ref_r if r not in args[ "ignore" ] ]
		del args[ "ignore" ]

	assert( len( args[ "models" ] ) > 1 )

	return args	

def read_pdb(
		pdbfile : str,
		res_list : List[ int ] = [],
		atoms : List[ str ] = [ "CA" ]
	) -> Dict[ int, np.ndarray ]:
	r""" Reads a PDB file and returns XYZ coordinates for residues of interest.
	Note that if multiple atoms are declared, the width of the array will change
	to accomodate them. So for example if N, CA, and C atoms are passed, then 
	the output array will be 9 wide to accomodate the XYZ coordinates of each.

	Parameters
	----------
	pdbfile : Name and directory of PDB file of interest
	res_list : List of residues to use
	atoms : Atoms to get XYZ coordinates

	Returns
	----------
	res_xyz : Dictionary mapping residue to a numpy array of XYZ coordinates

	"""

	res_xyz = dict()

	isempty = len( res_list ) == 0

	# Iterate over each residue
	pdb = Bio.PDB.PDBParser().get_structure( "TEMP", pdbfile )
	for residue in pdb.get_residues():
		res = residue.get_id()[ 1 ]

		# Exit condition in case residue is not of interest
		if res not in res_list and len( res_list ) > 0:
			continue
		elif res in res_list and len( res_list ) > 0 :
			res_list.remove( res )

		# In case multiple atoms are of interest
		res_xyz[ res ] = np.zeros( ( 3 * len( atoms ) ) )
		for i, atom in enumerate( atoms ):
			res_xyz[ res ][ i * 3 : i * 3 + 3 ] = residue[ atom ].coord

	if not isempty and len( res_list ) > 0:
		absl.warning( "\n".join( ( "Some residues not read or absent from PDB!",
				"\tPDB file:\t{},".format( pdbfile ),
				"\tResidues: {}".format( " ".join( res_list ) )
			) ) )

	return res_xyz

def _pca(
		modelpath : str,
		res : List[ int ] = [],
		atoms : List[ str ] = [ "CA" ]
	) -> NoReturn:
	r""" Runs PCA on all PDB files in the provided path.
	NOTE: Does not check if the sequences are the same, only that their
	lengths match!

	Parameters
	----------
	modelpath : Directory with aligned PDBs to run PCA
	res : Residues to use

	Returns
	----------
	None

	"""

	models = [ x for x in os.listdir( modelpath ) if x.endswith( ".pdb" ) ]
	print( f"Found { len( models ) } models" )

	all_xyz = []

	# Read XYZ values of each PDB file
	for file in models:
		xyz_vals = read_pdb( os.path.join( modelpath, file ), res ).values()
		all_xyz.append( np.concatenate( list( xyz_vals ) ) )

	all_xyz = np.vstack( all_xyz )
	print( all_xyz.shape )

	pca = PCA( n_components = 2 )
	new_xy = pca.fit_transform( all_xyz )
	logging.info( "Explained variance of PCA:\n\t{}\n\t{}".format(
			pca.explained_variance_ratio_[ 0 ],
			pca.explained_variance_ratio_[ 1 ]
		) )

	for model, ( x, y ) in zip( models, new_xy ):
		logging.info( "\t{}: {:2f}, {:2f}".format( model, x, y ) )

def _main( outpath : str = "temp" ) -> NoReturn:
	r""" Main function for reading and writing outputs.
	During execution, several temporary files will be written
	in the current working directory. These will then be deleter
	during cleanup.

	Parameters
	----------
	outpath : Where all aligned PDBs are saved

	Returns
	----------
	None

	"""

	# Fetch arguments
	args = _args()

	if not os.path.isdir( outpath ):
		os.mkdir( outpath )

	# Iterate over templates and run structure-based alignment
	for model in args[ "models" ]:

		name = os.path.basename( model ).split( "." )[ 0 ]

		outname = os.path.join( outpath, f"aligned_{ name }.pdb" )

		if os.path.isfile( outname ):
			logging.debug( "{} exists, skipping alignment".format( outname ) )

		else:
			logging.debug( "Aligning {}; output to {}".format( model, outname ) )

			cmd = " ".join( (
					args[ "tmalign" ],
					model,
					args[ "ref" ],
					"-o",
					outname					
				) )

			if not args[ "verbose" ]:
				cmd += " > /dev/null"

			_print_and_run( logging.debug, cmd )

	_pca( outpath )

if __name__ == "__main__":
	_main( "temp" )



