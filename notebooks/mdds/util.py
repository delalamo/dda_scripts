import Bio.PDB

import numpy as np
import jax.numpy as jnp

from jax import jit
from jax import vmap

from typing import Dict, List, Tuple

# TODO: Static initializations wherever possible

def get_atoms(
		model : Bio.PDB.Structure,
		atom : str
	) -> Dict[ Tuple[ int, str ], np.array ]:
	r""" Fetch all atoms of a type from model
	
	Parameters
	----------
	model : BioPython Structure object
	atom : Atom of interest ("CA", "N", etc)

	Returns
	----------
	Dictionary mapping residues and chains to XYZ positions

	"""

	atoms = dict()
	for chain in model.get_chains():
		for res in chain.get_residues():
			tempatom = atom
			if res.resname == "GLY" and atom == "CB":
				tempatom = "HB"
			try:
				pair = ( res.get_id()[ 1 ], chain.get_id() )
				atoms[ pair ] = res[ tempatom ].get_vector().get_array()
			except:
				#problem wit glycines
				#print( f"Error reading chain { chain } residue { res }" )
				continue
	return atoms

def calc_clashes(
		res : int,
		chain : str,
		model : Bio.PDB.Structure,
		dist : float
	) -> List[ np.array ]:
	r""" Fetch all atoms of a type from model
	
	Parameters
	----------
	res : Residue for clash calculation
	chain : Chain of residue of interest
	model : BioPython Structure object
	dist : Cutoff value (angstroms) for clash calculation (12 recommended)

	Returns
	----------
	XYZ positions for all atoms within 12 A of residue alpha carbon

	"""

	xyz_ca = model[ 0 ][ chain ][ res ][ "CA" ].coord

	clashing_atoms = []
	for atom in model.get_atoms():
		if "H" in atom.name:
			continue
		if np.linalg.norm( atom.coord - xyz_ca ) < dist:
			clashing_atoms.append( atom.coord )
	return clashing_atoms

def get_positions(
		x : np.array,
		headers : List[ str ],
		residues : List[ Tuple[ int, str ] ]
	) -> Dict[ Tuple[ int, str ], List[ float ] ]:
	r""" Fetch all atoms of a type from model
	
	Parameters
	----------
	x : Parameter vector
	headers : Name of variables in parameter vector
	residues : List of residues with dummy atoms being modeled

	Returns
	----------
	Dictionary of residues to XYZ coordinates

	"""

	xyz = dict()

	for r, c in residues:
		xyz[ ( r, c ) ] = [
			x[ headers.index( f"{ r }_{ c }_x" ) ],
			x[ headers.index( f"{ r }_{ c }_y" ) ],
			x[ headers.index( f"{ r }_{ c }_z" ) ]
		]
	return xyz

def set_params(
		residues : List[ Tuple[ int, str ] ]
	) -> Dict[ str, float ]:
	r""" Initialize parameters
	
	Parameters
	----------
	residues : List of residues with dummy atoms being modeled

	Returns
	----------
	Dictionary of initial XYZ coordinates with labels

	"""

	params = {}

	for r, c in residues:
		params[ f"{ r }_{ c }_x" ] = np.random.uniform( low=-25, high=25 )
		params[ f"{ r }_{ c }_y" ] = np.random.uniform( low=-25, high=25 )
		params[ f"{ r }_{ c }_z" ] = np.random.uniform( low=-25, high=25 )

	return params

@jit
def vdw(
		x1 : np.array,
		x2 : np.array
	) -> float:
	r""" Calculate VDW energy between two atoms using 6/12 potential
	
	Parameters
	----------
	x1 : XYZ of atom 1
	x2 : XYZ of atom 2

	Returns
	----------
	Lennard-Jones potential, assuming Emin = 0.05 and Rmin = 4

	"""

	d = jnp.linalg.norm( x1 - x2 )
	return -0.2 * ( ( 4. / d ) ** 12. - ( 4. / d ) ** 6 )

vdw_batch = vmap( vdw, in_axes=( None, 0 ) )

@jit
def xyz_energy_clash(
		xyz,
		ca,
		cb,
		n,
		neighbors
	):
	r""" Calculate VDW energy between two atoms using 6/12 potential
	
	Parameters
	----------
	xyz : XYZ of dummy atom
	ca : Alpha carbon vector
	cb : Beta carbon vector
	n : Backbone nitrogen vector
	neighbors : List of neighboring atoms (XYZ coords)

	Returns
	----------
	Negative log-energy of position (higher is more favorable)

	"""

	# Calculate distance
	cb = jnp.array( cb, dtype=jnp.float32 )
	neighbors = jnp.array( neighbors, dtype=jnp.float32 )
 
	ca = jnp.array( ca, dtype=jnp.float32 )
	xyz = jnp.array( xyz, dtype=jnp.float32 )
 
  
	e_clash = vdw_batch( xyz, neighbors ).sum()

	d = jnp.sqrt( ( ( xyz - ca ) ** 2 ).sum() )
	e_d = -0.5 * ( d - 8.0 ) ** 2

	# Calculate angle
	xyz_ca = xyz - ca
	cb_ca = cb - ca
 
	theta = jnp.dot( xyz_ca, cb_ca ) / (
		jnp.linalg.norm( xyz_ca ) * jnp.linalg.norm( cb_ca ) )
	e_theta = -1 * ( 0.9599 - theta ) ** 2 # in radians

	# Calculate dihedral N-CA-CB-ON

	xyz_cb = xyz - cb
	ca_cb = ca - cb
	n_ca = n - ca

	ca_cb_norm = ca_cb / jnp.linalg.norm( ca_cb )

	v = xyz_cb - jnp.dot( xyz_cb, ca_cb_norm ) * ca_cb_norm
	w = n_ca - jnp.dot( n_ca, ca_cb_norm ) * ca_cb_norm

	x = jnp.dot( v, w )
	y = jnp.dot( jnp.cross( ca_cb_norm, v ), w )

	phi = jnp.arctan2( y, x )

	e_phi = -1.9 * ( 1. + jnp.cos( phi - 4.1888 ) ) # in radians

	return e_clash + e_d + e_theta + e_phi


