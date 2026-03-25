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
```
usage: smiles_to_pdb [-h] --smiles SMILES --outpdb OUTPDB [--pH PH]
                     [--version]

Convert a SMILES string to a PDB file

options:
  -h, --help       show this help message and exit
  --smiles SMILES  Input SMILES string
  --outpdb OUTPDB  Output PDB file
  --pH PH          Target pH for protonation state
  --version        show program's version number and exit

```

#### prepare_protein
Prepare a protein structure for parameterization
               with Amber. This includes fixing residue names
               (e.g HIS to HID/HIE/HIP, CYS to CYX) and adding
               missing heavy atoms.
               Requires `pdb4amber` and `reduce` (from `AmberTools`)
               to be available

```
usage: prepare_protein [-h] -i INPDB -o OUTPDB

Prepare a PDB file for AMBER simulation.

options:
  -h, --help            show this help message and exit
  -i INPDB, --inpdb INPDB
                        Input PDB file
  -o OUTPDB, --outpdb OUTPDB
                        Output PDB file

```

#### het_param     
A workflow to parameterize heterogens (e.g. ligands)
               using `AmberTools`.
               Requires `antechamber` and `parmchk2` to be available.

```
usage: het_param [-h] --inpdb INPDB --het_name HET_NAME
                 [--het_charge HET_CHARGE] [--forcefield {gaff,gaff2}]
                 [--het_dir HET_DIR] [--no_opt] [--overwrite] [--version]

Parameterize a heterogen

options:
  -h, --help            show this help message and exit
  --inpdb INPDB         Input PDB file
  --het_name HET_NAME   Names of heterogen residues
  --het_charge HET_CHARGE
                        Formal charge of heterogen
  --forcefield {gaff,gaff2}
                        Force field to use
  --het_dir HET_DIR     Directory for heterogen files
  --no_opt              Skip optimization at QM stage
  --overwrite           Overwrite existing files
  --version             show program's version number and exit

  ```

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
```
usage: make_leap [-h] --inpdbs [INPDBS ...] --outinpcrd OUTINPCRD --outprmtop
                 OUTPRMTOP [--forcefields [FORCEFIELDS ...]]
                 [--het_names [HET_NAMES ...]] [--solvate {box,cube,oct}]
                 [--padding PADDING] [--het_dir HET_DIR]
                 [--ion_molarity ION_MOLARITY] [--version]

Generate tleap input script from PDB

options:
  -h, --help            show this help message and exit
  --inpdbs [INPDBS ...]
                        Input PDB file(s)
  --outinpcrd OUTINPCRD
                        Output AMBER .inpcrd file
  --outprmtop OUTPRMTOP
                        Output AMBER .prmtop file
  --forcefields [FORCEFIELDS ...]
                        Force fields to use
  --het_names [HET_NAMES ...]
                        Names of heterogen residues
  --solvate {box,cube,oct}
                        Type of water box to use
  --padding PADDING     minimum distance of solute atoms from box edge
  --het_dir HET_DIR     Directory for heterogen files
  --ion_molarity ION_MOLARITY
                        Target ionic strength (M)
  --version             show program's version number and exit
  ```

## Author

Charlie Laughton charles.laughton@nottingham.ac.uk
