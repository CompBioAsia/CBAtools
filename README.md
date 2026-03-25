# CBA-Tools
A package providing useful tools for Amber-focussed MD simulation setup.

Prepared for CompBioAsia 2026.

Installing the package makes available a number of command-line utilities:

#### smiles_to_pdb
Convert a SMILES string to a PDB file using OpenBabel
                and RDKit. This is useful for generating initial structures
                structures of ligands for MD simulations. The method
                attempts to generate structures in a protonation state
                appropriate for a chosen pH (default 7.4), but this is not
                guaranteed to be correct.

#### prepare_protein
Prepare a protein structure for parameterization
               with Amber. This includes fixing residue names
               (e.g HIS to HID/HIE/HIP, CYS to CYX) and adding
               missing heavy atoms.
               Requires `pdb4amber` and `reduce` (from `AmberTools`)
               to be available

#### het_param     
A workflow to parameterize heterogens (e.g. ligands)
               using `AmberTools`.
               Requires `antechamber` and `parmchk2` to be available.

#### make_leap
Generates a `tleap` input script to prepare input
               files (coordinates and topology/forcefield
               parameters) for MD simulation, from complete PDB
               format files of the solute components (e.g. all-atom
               models of protein plus ligand). Includes automatic
               parameterization of non-standard residues (using gaff
               or gaff2) if not already performed using `het_param`,
               and addition of water boxes and neutralizing counterions.
               The tool only works for non-covalent ligands (no bonds
               between the ligand and the protein).
               Requires `antechamber`, `parmchk2`, and `tleap` to be available.

## Author

Charlie Laughton charles.laughton@nottingham.ac.uk
