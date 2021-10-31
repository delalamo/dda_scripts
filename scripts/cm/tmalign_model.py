# Author: Diego del Alamo - diego.delalamo@gmail.com
# License: None

# This script threads a protein sequence onto a structure using
# TM-Align and Rosetta. 

import argparse
import os
import multiprocessing
import random

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

def _cm_options() -> str:
	r""" Small function for printing and calling a command

	Parameters
	----------
	None

	Returns
	----------
	opts : String with options for RosettaCM
	
	"""

	return " ".join( (
			
			# Relax options
			"-relax:minimize_bond_angles",
			"-relax:minimize_bond_lengths",
			"-relax:jump_move true",
			"-relax:min_type lbfgs_armijo_nonmonotone",
			"-default_max_cycles 200",

			# Reduce memory footprint
			 "-chemical:exclude_patches ",
			 	"LowerDNA",
			 	"UpperDNA",
			 	"Cterm_amidation",
			 	"SpecialRotamer",
			 	"VirtualBB",
			 	"ShoveBB",
			 	"VirtualDNAPhosphate",
			 	"VirtualNTerm",
			 	"CTermConnect",
			 	"sc_orbitals",
			 	"pro_hydroxylated_case1",
			 	"pro_hydroxylated_case2",
			 	"ser_phosphorylated",
			 	"thr_phosphorylated",
			 	"tyr_phosphorylated",
			 	"tyr_sulfated",
			 	"lys_dimethylated",
			 	"lys_monomethylated",
			 	"lys_trimethylated",
			 	"lys_acetylated",
			 	"glu_carboxylated",
			 	"cys_acetylated"
			 	"tyr_diiodinated",
			 	"N_acetylated",
			 	"C_methylamidated",
			 	"MethylatedProteinCterm"
		) )

def _args() -> Tuple[
		str, str, List[ str ], str, str, str, str, bool, bool, int, int
	]:
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
		xml : Name and location of XML file
		output_dir : Name of output PDB
		skip_s2 : Boolean of whether to skip the second stage
		verbose : Whether to print debug statements
	
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
			help=(
					"(required) Location of the Rosetta binaries "
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
			help="(optional) RosettaCM XML file",
			default=f"{ os.path.dirname( __file__ ) }/cm.xml",
			dest="xml"
		)

	parser.add_argument(
			'-o',
			'--output_prefix',
			metavar="PREFIX_",
			type=str,
			help=(
					"(optional) Prefix for output PDB. By default output "
					"names are identical to input names; default: None"
				),
			default="",
			dest="output"
		)

	parser.add_argument(
			'-s',
			'--skip_s2',
			help=(
					"(optional) Whether to skip the homology modeling"
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

	parser.add_argument(
			'-n',
			'--n_models',
			help="(optional) Number of models (default=1)",
			type=int,
			default=1,
			dest="n_models"
		)

	parser.add_argument(
			'--n_workers',
			help=(
					"(optional) Number of CPU cores for modeling (default=1)"
					"Set to -1 to use all available cores"
				),
			type=int,
			default=1,
			dest="n_workers"
		)

	parser.add_argument(
			'--max_templates',
			help=(
					"(optional) Maximum number of templates for modeling"
					"A value of 0 indicates all templates will be used"
					" (default=0)"
				),
			type=int,
			default=0,
			dest="max_templates"
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

def _cm(
		xml: str,
		fasta: str,
		rosetta: str,
		n_models: int,
		exe_type: str,
		extra_options: List[ str ]

	) -> NoReturn:
	r""" Uses Rosetta to close gaps and refine threaded models

	Parameters
	----------
	xml: XML file
	fasta : Reference FASTA filename
	rosetta : Rosetta path exe
	n_models : Number of models to generate
	n_workers : Number of CPU cores to use
	exe_type : Type of Rosetta executable to use

	Returns
	----------
	None

	"""



	return " ".join( (
			f"{ rosetta }/rosetta_scripts.{ exe_type }",
			f"-in:file:fasta { fasta }",
			f"-parser:protocol { xml }",
			f"-out:nstruct { n_models }",
			_cm_options(),
			"".join( extra_options )
		) )

def _write_xml(
		in_xml: str,
		lines: List[ str ],
		out_xml: str
	) -> NoReturn:
	r""" Generates an XML file for multi-template modeling

	Parameters
	----------
	in_xml: Input XML file
	lines : Template lines to add to XML file
	out_xml : Output XML file

	Returns
	----------
	None

	"""

	with open( out_xml, "w" ) as outfile:
		with open( in_xml ) as infile:
			for line in infile:
				if line.strip().startswith( "<Template" ):
					for l in lines:
						outfile.write( l )
				else:
					outfile.write( line )
	with open( out_xml ) as infile:
		for i, line in enumerate( infile ):
			logging.debug( "{}: {}".format( i, line.rstrip() ) )

def _setup_xml_all_templates(
		in_xml: str,
		pdbs: List[ str ],
		out_xml: str
	) -> NoReturn:
	r""" Generates an XML file for multi-template modeling

	Parameters
	----------
	in_xml: Input XML file
	names : List of templates
	out_xml : Output XML file

	Returns
	----------
	None

	"""

	pdb_lines = [ f"\t\t<Template pdb=\"{ p }.pdb\"/>\n" for p in pdbs ]
	_write_xml( in_xml, pdb_lines, out_xml )
	

def _setup_xml_subset(
		in_xml: str,
		n_pdbs: int,
		out_xml: str
	) -> NoReturn:
	r""" Generates an XML file for multi-template modeling

	Parameters
	----------
	in_xml: Input XML file
	names : List of templates
	out_xml : Output XML file

	Returns
	----------
	None

	"""

	pdb_lines = [ f"\t\t<Template pdb=%%m{ i }%%/>\n" for i in range( n_pdbs ) ]
	_write_xml( in_xml, pdb_lines, out_xml )

def _pick_pdbs(
		names: List[ str ],
		n_templates: int,
		allow_duplicates: bool=True
	) -> str:
	r""" Randomly picks PDB files and write command-line options

	Parameters
	----------
	names : PDB names to choose from (lacking ".pdb" suffix)
	n_templates : Number of templates to choose
	allow_duplicates : Whether same PDB can show up multiple times

	Returns
	----------
	cmd_option : String with command-line option for parser

	"""

	cmd_opt = [ "-parser:script_vars" ]
	while len( cmd_opt ) <= n_templates:
		rand_pdb = random.choice( names )
		if rand_pdb in cmd_opt and not allow_duplicates:
			continue
		cmd_opt.append( f"m{ len( cmd_opt ) - 1 }={ rand_pdb }.pdb" )
	return " ".join( cmd_opt )

def _main() -> NoReturn:
	r""" Main function for reading and writing outputs.
	During execution, several temporary files will be written
	in the current working directory. These will then be deleter
	during cleanup.

	Parameters
	----------
	None

	Returns
	----------
	None

	"""

	# Setting temporary names
	grishinfile = "temp.grishin"
	xmlfile = "temp.xml"

	exe_type="default.macosclangrelease"

	# Fetch arguments
	args = _args()

	if args[ "verbose" ]:
		logging.set_verbosity( logging.DEBUG )

	if args[ "n_workers" ] == -1:
		args[ "n_workers" ] = multiprocessing.cpu_count()

	if len( args[ "templates" ] ) < args[ "max_templates" ]:
		logging.warning( (
				"Found fewer templates than number specified by "
				"--max_templates. Automatically reducing the maximum "
				"number of templates to match this value."
			) )
		args[ "max_templates" ] = 0

	# Iterate over templates
	names = []
	for template in args[ "templates" ]:

		# Remove the directory and filetype from name
		names.append( os.path.basename( template ).split( "." )[ 0 ] )

		# Step 1a: Run structural alignment using TM-Align
		# Generates a grishin file for RosettaCM
		_tmalign_to_grishin(
				args[ "tmalign" ],
				args[ "pdb" ],
				template,
				grishinfile,
				names[ -1 ]
			)

		# Step 1b: Thread the sequence of interest onto the template
		_thread(
				args[ "rosetta" ],
				args[ "fasta" ],
				template,
				grishinfile,
				names[ -1 ],
				exe_type	
			)

	# If only a subset of templates should be used
	# Run the command once

	cmds = []
	if args[ "max_templates" ] == 0:

		_setup_xml_all_templates(
				args[ "xml" ],
				names,
				xmlfile
			)

		# Step 3: Run RosettaCM
		for i in range( args[ "n_workers" ] ):

			n_jobs = args[ "n_models" ] // args[ "n_workers" ]
			if args[ "n_models" ] % args[ "n_workers" ] > i:
				n_jobs += 1
			cmd = _cm(
					xmlfile,
					args[ "fasta" ],
					args[ "rosetta" ],
					n_jobs,
					exe_type,
					[ f"-out:prefix { i }_" ]
				)
			cmds.append( cmd )
			logging.debug( "{} {}".format( i, cmd ) )
	else:
		_setup_xml_subset(
				args[ "xml" ],
				args[ "max_templates" ],
				xmlfile
			)

		for i in range( args[ "n_models" ] ):
			cmd = _cm(
					xmlfile,
					args[ "fasta" ],
					args[ "rosetta" ],
					1,
					exe_type,
					[
							_pick_pdbs( names, args[ "max_templates" ] ),
							f"-out:prefix { i }_"
						]
				)
			cmds.append( cmd )
			logging.debug( "{} {}".format( i, cmd ) )


	# Step 5: Clean up
	for pdb in names:
		_print_and_run( logging.debug, f"rm { pdb }.pdb" )
	_print_and_run( logging.debug, f"rm { xmlfile }" )
	_print_and_run( logging.debug, f"rm { grishinfile }" )
	
if __name__ == "__main__":
	_main()


