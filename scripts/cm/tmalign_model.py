# Author: Diego del Alamo - diego.delalamo@gmail.com
# License: None

# This script threads a protein sequence onto a structure using
# TM-Align and Rosetta. 

import argparse
import os
import re

from absl import logging

from typing import Callable, List, NoReturn, Tuple

def _print_and_run(
		fn: Callable[ [ str ], None ],
		cmd: str
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

def _args() -> Tuple[ str, str, List[ str ], str, str, str, str, bool, bool ]:
	r""" Process input arguments for alignment
	
	Arguments defined with:
		--fasta Fasta file
		--pdb Reference PDB file
		--templates List of reference PDB files
		--rosetta Rosetta directory
		--tmalign Template PDBs
		--xml RosettaCM XML file
		--output Location of output PDBs
		--skip_s2 Whether to skip the second stage of modeling (RosettaCM)
		--verbose Whether to print all debug statements

	Parameters
	----------
	None

	Returns
	----------
	vals : Dictionary containing the following elements
		fasta : Location and name of fasta file
		ref_pdb : Location and name of reference PDB
		templates: Directory containing reference PDBs for threading
		rosetta_bin : Location of Rosetta (link to main/source/bin)
		tmalign_exe : Location of TM-Align executable
		output_dir : Name of outpud PDB
		skip_s2 : Boolean of whether to skip the second stage
	"""

	parser = argparse.ArgumentParser(
			prog="tmalign_model.py",
			description= (
					"This program does a bare-bones threading and "
					"homology modeling onto PDB templates of interest. "
					"Fine-tuning can be achieved by modifying the "
					"accompanying cm.xml file."
				),
			epilog=""
		)

	parser.add_argument(
			'-f',
			'--fasta',
			metavar="example.fasta",
			type=str,
			help="(required) FASTA file format for input sequence",
			required=True,
			dest="fasta"
		)

	parser.add_argument(
			'-p',
			'--pdb',
			metavar="example.pdb",
			type=str,
			help="(required) Reference PDB for structural alignment",
			required=True,
			dest="pdb"			
		)

	parser.add_argument(
			'-i',
			'--input_templates',
			metavar="template1.pdb template2.pdb ... templateN.pdb",
			type=str,
			nargs="+",
			help="(required) List of input template PDBs for threading",
			required=True,
			dest="templates"
		)

	parser.add_argument(
			'-r',
			'--rosetta',
			metavar="/path/to/rosetta/main/source/bin/",
			type=str,
			help=(	"(required) Location of the Rosetta binaries "
					"(main/source/bin)"
				),
			required=True,
			dest="rosetta"
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
			'-x',
			'--xml',
			metavar="cm.xml",
			type=str,
			help="(required) RosettaCM XML file",
			required=True,
			dest="xml"
		)

	parser.add_argument(
			'-o',
			'--output_prefix',
			metavar="PREFIX_",
			type=str,
			help=( 	"(optional) Prefix for output PDB. By default output "
					"names are identical to input names; default: None"
				),
			default="",
			dest="output"
		)

	parser.add_argument(
			'-s',
			'--skip_s2',
			help=(	"(optional) Whether to skip the homology modeling"
					"stage (threading only); default: False"
				),
			action="store_true",
			dest="skip_s2"
		)

	parser.add_argument(
			'-v',
			'--verbose',
			help="(optional) Set verbosity",
			action="store_true",
			dest="verbose"
		)

	args = parser.parse_args()

	return { arg: getattr( args, arg ) for arg in vars( args ) }

def _tmalign_to_grishin(
		tmalign: str,
		pdb1: str,
		pdb2: str,
		grishinfile : str,
		name: str
	) -> NoReturn:
	r""" Perform a structural alignment of reference to template

	Parameters
	----------
	tmalign : TM-Align exe
	pdb1 : Reference PDB
	pdb2 : Target PDB
	grishinfile : Filename for grishin file
	name : Template name

	Returns
	----------
	None
	"""

	cmd = " ".join( ( tmalign, pdb1, pdb2 ) )
	logging.debug( cmd )

	# Print command here
	tmalign_out = os.popen( cmd ).read().splitlines()

	for i, line in enumerate( tmalign_out ):
		logging.debug( "{}: {}".format( i, line.strip() ) )

	# Print the output here if absl.logging is in debug
	assert len( tmalign_out ) > 20

	# Write grishin file
	with open( grishinfile, "w" ) as outfile:
		outfile.write( f"## temp { name }\n" )
		outfile.write( "#\nscores_from_program: 0\n" )
		outfile.write( f"0 { tmalign_out[ 19 ].rstrip() }\n" )
		outfile.write( f"0 { tmalign_out[ 21 ].rstrip() }\n" )

	# Read grishin file
	with open( grishinfile ) as infile:
		for i, line in enumerate( infile ):
			logging.debug( "{}: {}".format( i, line.strip() ) )

def _thread(
		rosetta: str,
		fasta: str,
		template: str,
		grishinfile: str,
		name: str,
		exe_type: str
	) -> NoReturn:
	r""" Uses Rosetta to thread the sequence of one protein onto
	structure of another

	Parameters
	----------
	rosetta : Rosetta path exe
	fasta : Reference FASTA filename
	template : Template PDB for threading
	grishinfile : Filename for grishin file
	name : Name of the template
	exe_type : Type of Rosetta executable to use

	Returns
	----------
	None

	"""

	cmd = " ".join( ( 	f"{ rosetta }/partial_thread.{ exe_type }",
						f"-in:file:fasta { fasta }",
						f"-in:file:template_pdb { template }",
						f"-in:file:alignment { grishinfile }",
		) )

	_print_and_run( logging.debug, cmd )

	_print_and_run( logging.debug, f"rm { grishinfile }" )

def _cm(
		xml: str,
		fasta: str,
		rosetta: str,
		name: str,
		exe_type: str
	) -> NoReturn:
	r""" Uses Rosetta to close gaps and refine threaded models

	Parameters
	----------
	xml: XML file
	fasta : Reference FASTA filename
	rosetta : Rosetta path exe
	name : Name of the template
	exe_type : Type of Rosetta executable to use

	Returns
	----------
	None

	"""

	cmd = " ".join( (
			f"{ rosetta }/rosetta_scripts.{ exe_type }",
			f"-in:file:fasta { fasta }",
			f"-parser:protocol { xml }",
			f"-parser:script_vars model={ name }.pdb",
			f"-out:prefix { name }_"
		) )
	_print_and_run( logging.debug, cmd )

def _main() -> NoReturn:

	# Setting temporary names
	grishinfile = "temp.grishin"
	pdbfile = "temp.pdb"
	exe_type="default.macosclangrelease"

	# Fetch arguments
	args = _args()

	if args[ "verbose" ]:
		logging.set_verbosity( logging.DEBUG )

	# Iterate over templates
	
	for template in args[ "templates" ]:

		# Remove the directory and filetype from name
		name = os.path.basename( template ).split( "." )[ 0 ]

		# Step 1: Run structural alignment using TM-Align
		# Generates a grishin file for RosettaCM
		_tmalign_to_grishin(
				args[ "tmalign" ],
				args[ "pdb" ],
				template,
				grishinfile,
				name
			)

		# Step 2: Thread the sequence of interest onto the template
		_thread(
				args[ "rosetta" ],
				args[ "fasta" ],
				template,
				grishinfile,
				name,
				exe_type=exe_type				
			)

		# Step 3: Run RosettaCM
		_cm(
				args[ "xml" ],
				args[ "fasta" ],
				args[ "rosetta" ],
				name,
				exe_type=exe_type
			)
	
if __name__ == "__main__":
	_main()


