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

	parser.add_argument(
			'--cutoff',
			type=float,
			default=0.0,
			help="(optional) Set TM-score cutoff for alignment",
			dest="cutoff"
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

	if args[ "ignore" ] is not None:
		if args[ "keep" ] is not None:
			logging.warning( "Both --res_to_keep and --res_to_ignore options "
					"used! Ignoring --res_to_ignore."
				)
		else:
			ref_r = list( read_pdb( args[ "ref" ] ).keys() )
			args[ "keep" ] = [ r for r in ref_r if r not in args[ "ignore" ] ]
		del args[ "ignore" ]

	elif args[ "keep" ] is None:
		args[ "keep" ] = list( read_pdb( args[ "ref" ] ).keys() )

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
		logging.warning( "\n".join( ( "Some residues not read or absent from PDB!",
				"\tPDB file:\t{},".format( pdbfile ),
				"\tResidues: {}".format( " ".join( map( str, res_list ) ) )
			) ) )

	return res_xyz

def _pca(
		modelpath : str,
		model_tms : Dict[ str, float ],
		res : List[ int ] = [],
		atoms : List[ str ] = [ "CA" ]
	) -> NoReturn:
	r""" Runs PCA on all PDB files in the provided path.
	NOTE: Does not check if the sequences are the same, only that their
	lengths match!

	Parameters
	----------
	modelpath : Directory with aligned PDBs to run PCA
	model_tms : Dictionary of TMscores for each model
	res : Residues to use
	atoms : Atoms to use (default: CA only)

	Returns
	----------
	None

	"""

	models = model_tms.keys()
	print( f"Found { len( models ) } models" )

	all_xyz = []

	# Read XYZ values of each PDB file
	for file in models:
		xyz_vals = read_pdb( os.path.join( modelpath, file ), res, atoms ).values()
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
		if model not in model_tms:
			out = ",".join( map( str, ( model, x, y, 0.0 ) ) )
		else:
			out = ",".join( map( str, ( model, x, y, model_tms[ model ] ) ) )
		logging.info( "\t" + out )

def calc_tmscore(
		tmalign : str,
		pdb1 : str,
		pdb2 : str
	) -> float:
	r""" Executes TM-Align to calculate TM-score; does not align.
	TODO: This function shares code with scripts/cm/tmalign_model.py
	In the future I need to combine these two.

	Parameters
	----------
	tmalign : Location of executable
	pdb1 : First model to align
	pdb2 : Second model to align

	Returns
	----------
	TM-score (float ranging from 0 to 1)
	"""

	cmd = " ".join( ( tmalign, pdb1, pdb2 ) )
	logging.debug( cmd )

	# Print command here
	tms = []
	for i, line in enumerate( os.popen( cmd ).read().splitlines() ):
		logging.debug( "{}: {}".format( i, line.strip() ) )
		sl = line.split()
		
		# Failsafe in case the line is empty
		if len( sl ) >= 2:
			if sl[ 0 ] == "TM-score=":
				tms.append( float( sl[ 1 ] ) )
	return max( tms )

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

	model_tms = {}

	# Iterate over templates and run structure-based alignment
	for model in args[ "models" ]:

		name = "aligned_" + os.path.basename( model )
		outname = os.path.join( outpath, name )

		logging.debug( "Aligning {}; output to {}".format( model, outname ) )

		tm = calc_tmscore( args[ "tmalign" ], model, args[ "ref" ] )
		if tm > args[ "cutoff" ]:
			model_tms[ name ] = tm
			cmd = " ".join( (
				args[ "tmalign" ],
				model,
				args[ "ref" ],
				"-o",
				outname					
			) )

			if not args[ "verbose" ]:
				cmd += " > /dev/null"

			if not os.path.exists( outname ):
				_print_and_run( logging.debug, cmd )
			else:
				logging.debug( "Already found model" )
				
	_pca( outpath, model_tms, args[ "keep" ] )

if __name__ == "__main__":
	_main( "temp" )



