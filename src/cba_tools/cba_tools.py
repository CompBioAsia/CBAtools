# A set of utility tools for the CompBioAsia Molecular
# Dynamics tutorials.
#
# These Python functions provide an easy interface to
# a range of third party tools that are useful for the
# preparartion of molecular systems for MD simulation.
# These include:
#
# a) Packages that must be installed/available:
#   Blast
#   AmberTools
#
# b) Python packages that will be automatically installed:
#   MDTraj
#   OpenMM
#   alphafix
#
# The functions are:
#
#   smiles_to_pdb: Convert a SMILES string to a PDB file using OpenBabel
#                  and RDKit. This is useful for generating initial structures
#                  structures of ligands for MD simulations. The method
#                  attempts to generate structures in a protonation state
#                  appropriate for a chosen pH (default 7.4), but this is not
#                  guaranteed to be correct.
#
#   prepare_protein: Prepare a protein structure for parameterization
#                  with Amber. This includes fixing residue names
#                  (e.g HIS to HID/HIE/HIP, CYS to CYX) and adding
#                  missing heavy atoms.
#                  Requires pdb4amber and reduce to be available
#
#   het_param:     A workflow to parameterize heterogens (e.g. ligands)
#                  using AmberTools.
#                  Requires antechamber and parmchk2 to be available.
#
#   param:         A complete AMBER-focussed workflow to prepare input
#                  files (coordinates and topology/forcefield
#                  parameters) for MD simulation, from complete PDB
#                  format files of the solute components (e.g. all-atom
#                  models of protein plus ligand). Includes automatic
#                  parameterization of non-standard residues (using gaff
#                  or gaff2) if not already performed using het_param,
#                  and addition of water boxes and neutralizing counterions.
#                  The tool only works for non-covalent ligands (no bonds
#                  between the ligand and the protein).
#                  Requires antechamber, parmchk2, and tleap to be available.
#
#
#
# Be aware that all these workflows can be confused by unusual
# or in some way particularly awkward systems (e.g. bad initial
# coordinates).
#
# SO PLEASE ALWAYS CHECK THE RESULTS CAREFULLY!
#

import mdtraj as mdt
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from openbabel import openbabel as ob

from crossflow.tasks import SubprocessTask, CalledProcessError
from crossflow.filehandling import FileHandler, FileHandle
from functools import cache
from enum import IntEnum
import shutil

from pathlib import Path
import sys


#  Part 1: Various utilities

def _aliased(cmd):
    '''
    Little utility to see if a comand is aliased
    '''
    alias = SubprocessTask('alias {cmd}')
    alias.set_inputs(['cmd'])
    alias.set_outputs(['STDOUT'])
    al = alias(cmd)
    return al


def _check_available(cmd):
    '''
    Little utility to check a required command is available

    '''
    if shutil.which(cmd) is None:
        if _aliased(cmd) == '':
            raise FileNotFoundError(f'Error: cannot find the {cmd} command')


def _check_exists(filename):
    '''
    Little utility to check if a required file is present

    '''
    if not Path(filename).exists():
        raise FileNotFoundError(f'Error: cannot find required file {filename}')


def _check_overwrite(path, overwrite):
    if path.exists():
        if overwrite:
            print(f'Warning, existing file {path} will be over-written',
                  file=sys.stderr)
        else:
            raise FileExistsError(f'Error: {path} already exists')


def _pdbify(prot_in):
    '''
    Convert something appropriate to PDB format

    Return as a FileHandle
    '''

    fh = FileHandler()
    if isinstance(prot_in, mdt.Trajectory):
        tmppdb = fh.create('tmp.pdb')
        tmppdb.write_text('')  # to sidestep a bug
        prot_in.save(tmppdb)
        pdbout = fh.load(tmppdb)
    elif isinstance(prot_in, (str, Path)):
        pdbout = fh.load(prot_in)
    elif isinstance(prot_in, FileHandle):
        pdbout = prot_in
    else:
        raise TypeError(f'Unsupported input type {type(prot_in)}')
    return pdbout


def _trajify(prot_in, standard_names=False):
    '''
    Convert something appropriate to MDTraj trajectory format
    '''
    if not isinstance(prot_in, (str, Path, mdt.Trajectory, FileHandle)):
        raise TypeError(f'Unsupported input type {type(prot_in)})')

    if isinstance(prot_in, (str, Path)):
        _check_exists(prot_in)

    if not isinstance(prot_in, mdt.Trajectory):
        # Convert to MDTraj trajectory
        prot_in = mdt.load_pdb(prot_in, standard_names=standard_names,
                               no_boxchk=True)
    return prot_in


def unique_chain_ids(t):
    '''
    Return a copy of the trajectory with chain ids set.

    Deals with the situation of PDB files with several
    chains, but no chain ids set.

    Chains with existing names (not ' '), are preserved.
    '''
    t = _trajify(t)
    t = mdt.Trajectory(t.xyz.copy(), t.topology.copy())
    chain_names = [c.chain_id for c in t.topology.chains]
    if ' ' not in chain_names:
        return t
    chain_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    j = 0
    for i, chain in enumerate(t.topology.chains):
        if chain.chain_id == ' ':
            # Assign a new chain id
            while chain_letters[j] in chain_names:
                j += 1
                if j >= len(chain_letters):
                    print('Warning: too many chains in the trajectory',
                          file=sys.stderr)
                    return t
            chain.chain_id = chain_letters[j]
            chain_names[i] = chain.chain_id
    return t


def bumps(prot_in, cutoff=0.2):
    '''
    Report close contacts in a protein structure

    Args:
        prot_in: The input protein structure (PDB file or MDTraj trajectory)
        cutoff: Distance cutoff for defining close contacts (default: 0.2 nm)

    Returns:
        str: A report of close contacts in the protein structure

    '''
    prot_in = _trajify(prot_in)
    contacts = []
    for i in range(prot_in.topology.n_residues - 2):
        for j in range(i+2, prot_in.topology.n_residues):
            contacts.append([i, j])
    c = mdt.compute_contacts(prot_in, contacts)
    d = c[0][0]

    result = ''
    for i in np.argsort(d):
        if d[i] < 0.2:
            ra = str(prot_in.topology.residue(contacts[i][0]))
            rb = str(prot_in.topology.residue(contacts[i][1]))
            result += f'{ra:6s} - '
            result += f'{rb:6s} {d[i]:.3f}\n'
    return result


def hetify(pdbin):
    '''
    Change ATOM to HETATM for non-protein atoms

    Note: this function doesn't understand nucleic acids.

    '''
    t = _trajify(pdbin)
    non_protein = t.topology.select('not protein')
    het_residues = set([t.topology.atom(i).residue.name for i in non_protein])
    for r in ('HID', 'HIE', 'HIP', 'CYX', 'ASH', 'GLH', 'CYM'):
        if r in het_residues:
            het_residues.remove(r)
    in_data = _pdbify(pdbin).read_text().split('\n')
    out_data = ''
    for line in in_data:
        if line[:5] == 'ATOM ':
            resname = line[17:20]
            if resname in het_residues:
                line = 'HETATM' + line[6:]
        out_data += line + '\n'
    fh = FileHandler()
    out_pdb = fh.create('tmp.pdb')
    out_pdb.write_text(out_data)
    return out_pdb


def residue_id(r):
    '''
    Generate a unique identifier string for a residue
    based on its name, sequence number, and chain ID.

    e.g. ALA12.A
    '''
    return f"{r.name}{r.resSeq}.{r.chain.chain_id}"


def atom_id(a):
    '''
    Generate a unique identifier string for an atom
    based on its residue and atom name.

    e.g. ALA12.A@CA
    '''

    return f"{residue_id(a.residue)}@{a.name}"


# For the Smith-Waterman code:
class Score(IntEnum):
    MATCH = 1
    MISMATCH = -1
    GAP = -1


# Assigning the constant values for the traceback
class Trace(IntEnum):
    STOP = 0
    LEFT = 1
    UP = 2
    DIAGONAL = 3


def smith_waterman(seq1, seq2):
    '''A simple Smith-Waterman local alignment routine.

    Adapted from:
    https://github.com/slavianap/Smith-Waterman-Algorithm/blob/master/Script.py

    Args:
       seq1 (str): The first sequence
       seq2 (str): The second sequence

    Returns:
        tuple: The aligned sequences
    '''
    # Generating the empty matrices for storing scores and tracing
    row = len(seq1) + 1
    col = len(seq2) + 1
    matrix = np.zeros(shape=(row, col), dtype=int)
    tracing_matrix = np.zeros(shape=(row, col), dtype=int)

    # Initialising the variables to find the highest scoring cell
    max_score = -1
    max_index = (-1, -1)

    # Calculating the scores for all cells in the matrix
    for i in range(1, row):
        for j in range(1, col):
            # Calculating the diagonal score (match score)
            match_value = Score.MATCH if seq1[i - 1] == seq2[j - 1] \
                            else Score.MISMATCH
            diagonal_score = matrix[i - 1, j - 1] + match_value

            # Calculating the vertical gap score
            vertical_score = matrix[i - 1, j] + Score.GAP

            # Calculating the horizontal gap score
            horizontal_score = matrix[i, j - 1] + Score.GAP

            # Taking the highest score
            matrix[i, j] = max(0, diagonal_score, vertical_score,
                               horizontal_score)

            # Tracking where the cell's value is coming from
            if matrix[i, j] == 0:
                tracing_matrix[i, j] = Trace.STOP

            elif matrix[i, j] == horizontal_score:
                tracing_matrix[i, j] = Trace.LEFT

            elif matrix[i, j] == vertical_score:
                tracing_matrix[i, j] = Trace.UP

            elif matrix[i, j] == diagonal_score:
                tracing_matrix[i, j] = Trace.DIAGONAL

            # Tracking the cell with the maximum score
            if matrix[i, j] >= max_score:
                max_index = (i, j)
                max_score = matrix[i, j]

    # Initialising the variables for tracing
    aligned_seq1 = ""
    aligned_seq2 = ""
    current_aligned_seq1 = ""
    current_aligned_seq2 = ""
    (max_i, max_j) = max_index

    # Tracing and computing the pathway with the local alignment
    while tracing_matrix[max_i, max_j] != Trace.STOP:
        if tracing_matrix[max_i, max_j] == Trace.DIAGONAL:
            current_aligned_seq1 = seq1[max_i - 1]
            current_aligned_seq2 = seq2[max_j - 1]
            max_i = max_i - 1
            max_j = max_j - 1

        elif tracing_matrix[max_i, max_j] == Trace.UP:
            current_aligned_seq1 = seq1[max_i - 1]
            current_aligned_seq2 = '-'
            max_i = max_i - 1

        elif tracing_matrix[max_i, max_j] == Trace.LEFT:
            current_aligned_seq1 = '-'
            current_aligned_seq2 = seq2[max_j - 1]
            max_j = max_j - 1

        aligned_seq1 = aligned_seq1 + current_aligned_seq1
        aligned_seq2 = aligned_seq2 + current_aligned_seq2

    # Reversing the order of the sequences
    aligned_seq1 = aligned_seq1[::-1]
    aligned_seq2 = aligned_seq2[::-1]

    #  Add unmatched ends, if any:
    if max_i > 0 and max_j > 0:
        n_extra = min(max_i, max_j)
        aligned_seq1 = seq1[max_i-n_extra:max_i] + aligned_seq1
        aligned_seq2 = seq2[max_j-n_extra:max_j] + aligned_seq2
        max_i = max_i - n_extra
        max_j = max_j - n_extra

    if max_i > 0:
        aligned_seq1 = seq1[:max_i] + aligned_seq1
        aligned_seq2 = '-' * max_i + aligned_seq2
    elif max_j > 0:
        aligned_seq2 = seq2[:max_j] + aligned_seq2
        aligned_seq1 = '-' * max_j + aligned_seq1

    if max_index[0] < row - 1:
        aligned_seq1 += seq1[max_index[0]:]
        aligned_seq2 += '-' * (row - 1 - max_index[0])
    elif max_index[1] < col - 1:
        aligned_seq2 += seq2[max_index[1]:]
        aligned_seq1 += '-' * (col - 1 - max_index[1])
    return aligned_seq1, aligned_seq2


def aln_score(alignment):
    '''
    Calculate the number of matches, mismatches and gaps
        in a pairwise alignment.

    Args:
        alignment (tuple): A tuple containing two aligned sequences.

    Returns:
        tuple: A tuple containing the number of matches, mismatches, and gaps.

    '''
    if not len(alignment[0]) == len(alignment[1]):
        raise ValueError('Error: alignments must be the same length')
    matches = 0
    mismatches = 0
    gaps = 0
    for a, b in zip(*alignment):
        if '-' in a+b and '--' not in a+b:
            gaps += 1
        else:
            if a == b:
                matches += 1
            else:
                mismatches += 1
    return matches, mismatches, gaps

#  Part 2: Sequence-based manipulation of "pure" protein structures
#          represented as single-snapshot MDTrajectory files


def match_align(pdb_in, pdb_ref, cutoff=0.02, renumber=False, align=True):
    '''
    Superimpose pdb_in onto pdb_ref, based on sequence alignment

    The C-alpha atoms used for least-squares fitting are iteratively pruned
    until all pairs are within <cutoff> nanometers.

    Args:
        pdb_in: The input PDB file or MDTraj trajectory for the structure
                to align.
        pdb_ref: The reference PDB file or MDTraj trajectory for the structure
                to align to.
        cutoff: The distance cutoff for defining close contacts
                (default: 0.02 nm).
        renumber: If True, the returned PDB file has its residue sequence
                  numbers changed to match those in the reference PDB.
        align: If False, no structure superposition is performed (only useful
               if renumber=True!).

   Returns:
        Filehandle: The path to the aligned PDB file.

    '''
    t_in = _trajify(pdb_in)
    t_ref = _trajify(pdb_ref)

    seq_in = t_in.topology.to_fasta()
    seq_ref = t_ref.topology.to_fasta()
    if len(seq_in) != len(seq_ref):
        raise ValueError(
            'Error: input and reference sequences must have'
            ' the same number of chains')

    n_chains = len(seq_in)
    alignment = ["", ""]
    for i in range(n_chains):
        aln = smith_waterman(seq_in[i], seq_ref[i])
        alignment[0] += aln[0]
        alignment[1] += aln[1]

    i = -1
    j = -1
    pairs = []
    ca_in = t_in.topology.select('name CA')
    ca_ref = t_ref.topology.select('name CA')
    pair_pos = {}
    k = 0
    for a, b in zip(*alignment):
        if a != '-':
            i += 1
        if b != '-':
            j += 1
        if '-' not in a+b:
            pair = (ca_in[i], ca_ref[j])
            pairs.append(pair)
            pair_pos[pair] = k
        k += 1

    t_copy = mdt.Trajectory(t_in.xyz.copy(), t_in.topology.copy())
    tt_copy = t_copy.topology
    tt_ref = t_ref.topology
    if renumber:
        for pair in pair_pos:
            seq = tt_ref.atom(pair[1]).residue.resSeq
            tt_copy.atom(pair[0]).residue.resSeq = seq

    if not align:
        return _pdbify(t_copy), alignment

    unconverged = True
    # Iteratively remove pairs that are too far apart
    while unconverged:
        atom_indices = [p[0] for p in pairs]
        ref_atom_indices = [p[1] for p in pairs]
        t_out = t_copy.superpose(t_ref, atom_indices=atom_indices,
                                 ref_atom_indices=ref_atom_indices)
        dx = t_out.xyz[0, atom_indices] - t_ref.xyz[0, ref_atom_indices]
        err = np.linalg.norm(dx, axis=1)
        if err.max() > cutoff:
            ierr = np.argsort(err)
            icut = np.argmax(err[ierr] > cutoff)
            lerr = len(err)
            r1 = (lerr-icut) // 2
            r2 = lerr // 10
            r = min(r1, r2)
            r = max(r, 1)
            discards = ierr[-r:]
            old_pairs = pairs
            pairs = []
            for i, p in enumerate(old_pairs):
                if i not in discards:
                    pairs.append(p)
        else:
            unconverged = False
    pairings = [' '] * len(alignment[0])
    for p in pairs:
        pairings[pair_pos[p]] = '|'
    alignment = (alignment[0],  ''.join(pairings), alignment[1])
    return _pdbify(t_out), alignment


def clean_pdb(pdbfile):
    '''
    Remove junk added to PDB files by reduce
    '''
    in_data = pdbfile.read_text().split('\n')
    out_data = []
    for line in in_data:
        words = line.split()
        if len(words) > 0:
            if words[0] in ('ATOM', 'HETATM', 'TER', 'END', 'CONECT'):
                out_data.append(line)
    pdbfile.write_text('\n'.join(out_data))


def complete(pdb_in):
    '''
    Complete a structure by adding missing atoms using pdb4amber.

    The difference from pdb4amber itself is that this version included NQH
    flips

    Args:
        pdb_in (path-like): input structure in PDB format

    Returns:
        Filehandle: The path to the completed PDB file.

    '''
    _check_available('pdb4amber')
    _check_available('reduce')
    # Step 1: add missing heavy atoms
    pdb4amber = SubprocessTask(
        'pdb4amber -i in.pdb --add-missing-atoms --no-conect -o out.pdb')
    pdb4amber.set_inputs(['in.pdb'])
    pdb4amber.set_outputs(['out.pdb'])
    out = pdb4amber(pdb_in)
    # Step 1b: correct His residue names
    out.write_text(out.read_text().replace('HIE', 'HIS'))

    # Step 2; remove all current hydrogens, then run through reduce, though its
    # only the NQH flips that interest us:
    trim = SubprocessTask(
        'reduce -Trim in.pdb > out.pdb'
    )
    trim.set_inputs(['in.pdb'])
    trim.set_outputs(['out.pdb', 'DEBUGINFO'])
    build = SubprocessTask(
        'reduce -BUILD in.pdb > out.pdb'
    )
    build.set_inputs(['in.pdb'])
    build.set_outputs(['out.pdb', 'DEBUGINFO'])

    out1, info1 = trim(out)
    clean_pdb(out1)
    if isinstance(info1, CalledProcessError):
        print("Warning: reduce returned with non-zero exit code",
              file=sys.stderr)
    out2, info2 = build(out1)
    clean_pdb(out2)
    if isinstance(info2, CalledProcessError):
        print("Warning: reduce returned with non-zero exit code",
              file=sys.stderr)
    out3, info3 = trim(out2)
    clean_pdb(out3)
    if isinstance(info3, CalledProcessError):
        print("Warning: reduce returned with non-zero exit code",
              file=sys.stderr)

    # Step 3: Correct the HIS residues
    pdb4amber = SubprocessTask(
        'pdb4amber -i in.pdb --reduce --no-conect -o out.pdb'
        )
    pdb4amber.set_inputs(['in.pdb'])
    pdb4amber.set_outputs(['out.pdb'])
    out = pdb4amber(out3)
    # Step 4: Remove hydrogens
    strip_h = SubprocessTask(
        'pdb4amber -i in.pdb --nohyd -o out.pdb')
    strip_h.set_inputs(['in.pdb'])
    strip_h.set_outputs(['out.pdb'])
    pdb_out = strip_h(out)
    print(pdb_diff(pdb_in, pdb_out))
    return hetify(pdb_out)


def pdb_diff(p_before, p_after):
    '''
    Compare two PDB files.

    Report differences in residue names, atom names, and coordinates.

    Args:
        p_before (str): Path to the PDB file before modification.
        p_after (str): Path to the PDB file after modification.

    Returns:
        str: A report of the differences between the two PDB files.

    '''
    t_before = _trajify(p_before, standard_names=False)
    t_after = _trajify(p_after, standard_names=False)
    report = ''
    for r1, r2 in zip(t_before.topology.residues, t_after.topology.residues):
        if r1.name != r2.name:
            report += f'{r1.name}{r1.resSeq}.{r1.chain.chain_id}: '
            report += f'modelled as {r2.name}\n'
        r1_atoms = [a.name for a in r1.atoms]
        r2_atoms = [a.name for a in r2.atoms]
        added_Heavy_atoms = [a for a in r2_atoms
                             if a not in r1_atoms and a[0] != 'H']
        if added_Heavy_atoms:
            report += f'{r1.name}{r1.resSeq}.{r1.chain.chain_id}: '
            report += f'Added heavy atoms: {", ".join(added_Heavy_atoms)}\n'

        if r1.name in ('HIS', 'ASN', 'GLN'):
            # Check for NQH flips
            if r1.name == 'HIS':
                a1 = [a.index for a in r1.atoms if a.name == 'ND1']
                a2 = [a.index for a in r2.atoms if a.name == 'ND1']
            elif r1.name == 'ASN':
                a1 = [a.index for a in r1.atoms if a.name == 'OD1']
                a2 = [a.index for a in r2.atoms if a.name == 'OD1']
            elif r1.name == 'GLN':
                a1 = [a.index for a in r1.atoms if a.name == 'OE1']
                a2 = [a.index for a in r2.atoms if a.name == 'OE1']
            if len(a1) > 0 and len(a2) > 0:
                if not np.allclose(t_before.xyz[0, a1],
                                   t_after.xyz[0, a2], atol=0.01):
                    report += f'{r1.name}{r1.resSeq}.{r1.chain.chain_id}: '
                    report += 'NQH flip\n'
    return report


#  Part 4: Tools for ligand parameterization with AmberTools

def smiles_to_pdb(smi, pH=7.0):
    '''
    Convert an input SMILES representation to a PDB file

    Args:
        smi (str): Input SMILES string
        pH (float): target pH

    Returns:
        pdb_out (str): The PDB file as a string
        charge (int): The formal charge on the molecule

    '''
    obc = ob.OBConversion()
    obc.SetInAndOutFormats('smi', 'smi')

    obmol = ob.OBMol()
    obc.ReadString(obmol, smi)
    obmol.CorrectForPH(pH)
    smi_pH = obc.WriteString(obmol)
    charge = smi_pH.count('+]') - smi_pH.count('-]')
    mol_pH = Chem.MolFromSmiles(smi_pH)
    mol_pH_H = Chem.AddHs(mol_pH)
    rdDistGeom.EmbedMolecule(mol_pH_H)
    pdb_out = Chem.MolToPDBBlock(mol_pH_H)

    return pdb_out, charge


@cache
def parameterize(source, residue_name, charge=0, gaff='gaff',
                 het_dir='.', no_opt=False, overwrite=False):
    '''
    Paramaterize a non-standard residue (heterogen)

    Uses antechamber and parmchk2 to generate .mol2 and .frcmod files.

    Args:
       source (str): the PDB file name
       residue_name (str): the three-letter residue code for the heterogen
       charge: the formal charge on the heterogen
         gaff (str): which version of gaff to use ('gaff' or 'gaff2')
         het_dir (str or Path): location of the directory to write
                                 heterogen parameters
        no_opt (bool): if True, do not perform energy minimization
                       on the heterogen structure before parameterization
        overwrite (bool): if True, overwrite any existing parameter files

    Returns:
       list [mol2, frcmod]: crossflow.FileHandles

    '''
    if gaff not in ('gaff', 'gaff2'):
        raise ValueError(f'Error: unrecognised gaff option "{gaff}": '
                         'must be "gaff" or "gaff2"')
    _check_exists(source)

    available = True
    for ext in ['.pdb', '.mol2', '.frcmod']:
        file = Path(het_dir) / f'{residue_name}{ext}'
        if not file.exists():
            available = False
    if available:
        print('No fresh parameterization required')
        return

    traj = mdt.load_pdb(source, standard_names=False)
    het_sel = traj.topology.select(f'resname {residue_name}')
    if len(het_sel) == 0:
        raise ValueError(f'Error: no residue {residue_name} found in {source}')

    # A trajectory that contains all copies of the selected heterogen:
    traj_hets = traj.atom_slice(het_sel)
    # A trajectory that contains a single copy of the selected heterogen:
    traj_het = traj_hets.atom_slice(traj_hets.topology.select('resid 0'))
    # Remove bonds as they cause problems
    traj_het.topology._bonds = []
    het_pdb = Path(het_dir) / f'{residue_name}.pdb'
    _check_overwrite(het_pdb, overwrite)
    traj_het.save(het_pdb)

    # Run antechamber
    _check_available('antechamber')
    ek = ''
    if no_opt:
        ek = "-ek 'qm_theory=\"AM1\", grms_tol=0.0005,"
        ek += " scfconv=1.d-10, maxcyc=0'"
        print('Skipping energy minimization of heterogen')
    else:
        print('Performing energy minimization of heterogen')
    if gaff == 'gaff':
        antechamber = SubprocessTask('antechamber -i infile.pdb -fi pdb'
                                     ' -o outfile.mol2 -fo mol2 -c bcc'
                                     ' -nc {charge} {ek}')
    else:
        antechamber = SubprocessTask('antechamber -i infile.pdb -fi pdb'
                                     ' -o outfile.mol2 -fo mol2 -c bcc'
                                     ' -nc {charge} {ek} -at gaff2')
    antechamber.set_inputs(['infile.pdb', 'charge', 'ek'])
    antechamber.set_outputs(['outfile.mol2', 'STDOUT'])
    outmol2, stdout = antechamber(traj_het, charge, ek)
    if outmol2 is None:
        raise RuntimeError(f'Error in antechamber:\n{stdout}')
    # run parmchk2
    _check_available('parmchk2')
    if gaff == 'gaff':
        parmchk = SubprocessTask('parmchk2 -i infile.mol2 -f mol2 -o'
                                 ' outfile.frcmod')
    else:
        parmchk = SubprocessTask('parmchk2 -s 2 -i infile.mol2 -f mol2'
                                 ' -o outfile.frcmod')
    parmchk.set_inputs(['infile.mol2'])
    parmchk.set_outputs(['outfile.frcmod'])
    frcmod = parmchk(outmol2)

    mol2file = Path(het_dir) / f'{residue_name}.mol2'
    _check_overwrite(mol2file, overwrite)
    outmol2.save(f'{residue_name}.mol2')

    frcmodfile = Path(het_dir) / f'{residue_name}.frcmod'
    _check_overwrite(frcmodfile, overwrite)
    frcmod.save(f'{residue_name}.frcmod')


#  Part 5: Tools for (Amber) MD simulation preparation

def make_leap(inpdbs, outinpcrd, outprmtop, het_names=None,
              het_dir='.',
              forcefields=None,
              solvate=None,
              ion_molarity=None,
              padding=10.0):
    """
    Generate a tleap script to prepare a system for MD simulation.

    Args:
        inpdbs (list): names of input PDB files
        outinpcrd (path-like): name of output inpcrd file
        outprmtop (path-like): name of output prmtop file
        het_names (None or list): 3-letter residue names for heterogens
        forcefields (None or list): List of forcefields to use
        solvate (None or str): Solvation option - can be 'box',
                               'cube', or 'oct'.
        padding (float): minimum distance from any solute atom
                        to a periodic box boundary (Angstroms)
        ion_molarity (float or None): If not None, add Na+ and Cl- ions
                                      to reach this molarity (M)
        het_dir (str): Directory to search for heterogen parameter files
    Returns:
        str: The tleap script as a string
    """
    for inpdb in inpdbs:
        _check_exists(inpdb)

    if not forcefields:
        print('Warning: no forcefields specified, '
              'defaulting to "protein.ff14SB"', file=sys.stderr)
        forcefields = ['protein.ff14SB']
    has_protein_ff = False
    for ff in forcefields:
        if 'protein' in ff:
            has_protein_ff = True
    if not has_protein_ff:
        if 'gaff2' in forcefields:
            print('Warning: no protein forcefield specified,'
                  ' defaulting to protein.ff19SB.', file=sys.stderr)
            forcefields.append('protein.ff19SB')
        else:
            print('Warning: no protein forcefield specified,'
                  ' defaulting to protein.ff14SB.', file=sys.stderr)
            forcefields.append('protein.ff14SB')
    if solvate:
        if solvate not in ['oct', 'box', 'cube']:
            raise ValueError(f'Error: unrecognised solvate option "{solvate}"')
        water_ff = False
        for ff in forcefields:
            if 'water' in ff:
                water_ff = True
        if not water_ff:
            print('Warning: no water forcefield specified but'
                  ' solvation required.', file=sys.stderr)
            if 'protein.ff19SB' in forcefields:
                print('Defaulting to "water.opc" forcefield.',
                      file=sys.stderr)
                forcefields.append('water.opc')
            else:
                print('Defaulting to "water.tip3p" forcefield.',
                      file=sys.stderr)
                forcefields.append('water.tip3p')

    if het_names is not None:
        if 'gaff' not in forcefields and 'gaff2' not in forcefields:
            print('Warning - heterogens are present but no gaff/gaff2 '
                  'forcefield has been specified.', file=sys.stderr)
            print('Will default to using "gaff".', file=sys.stderr)
            forcefields.append('gaff')

    if not ion_molarity:
        try:
            script = leap(
                inpdbs, forcefields, het_names=het_names,
                solvate=solvate, padding=padding, het_dir=het_dir,
                script_only=True)
            script = script.replace('system.prmtop', str(outprmtop))
            script = script.replace('system.inpcrd', str(outinpcrd))
            return script
        except RuntimeError as e:
            print(f'Error in leap:\n{e}')
            exit(1)

    try:
        prmtop, inpcrd, stdout = leap(
            inpdbs, forcefields, het_names=het_names,
            solvate=solvate, padding=padding, het_dir=het_dir)
    except RuntimeError as e:
        print(f'Error in leap:\n{e}')
        exit(1)
    if ion_molarity:
        ttmp = mdt.load(inpcrd, top=prmtop)
        n_waters = len(ttmp.topology.select('name O and resname HOH'))
        n_na = len(ttmp.topology.select('name "Na+"'))
        n_cl = len(ttmp.topology.select('name "Cl-"'))
        n_ions = int(ion_molarity * n_waters/55.56)
        if n_na > 0:
            n_cl = n_ions - n_na
            n_na = n_ions
        else:
            n_na = n_ions - n_cl
            n_cl = n_ions
        n_na = max(n_na, 0)
        n_cl = max(n_cl, 0)

        try:
            script = leap(
                inpdbs, forcefields, het_names=het_names,
                solvate=solvate, padding=padding, het_dir=het_dir,
                n_na=n_na, n_cl=n_cl, script_only=True)
            script = script.replace('system.prmtop', str(outprmtop))
            script = script.replace('system.inpcrd', str(outinpcrd))
            return script
        except RuntimeError as e:
            print(f'Error in leap:\n{e}')
            exit(1)


def leap(amberpdbs, ff, het_names=None, solvate=None, padding=10.0, het_dir='.',
         n_na=0, n_cl=0, script_only=False):
    '''
    Parameterize a molecular system using tleap.

    Args:
       amberpdbs (list): List of Amber-compliant PDB files
       ff (list): The force fields to use.
       het_names (list): List of parameterised heterogens
       solvate (str or None): type of periodic box ('box', 'cube', or 'oct')
       padding (float): Clearance between solute and any box edge (Angstroms)
       het_dir (str or Path): location of the directory containing heterogen
                              parameters
       n_na (int): number of Na+ ions to add (0 = minimal salt)
       n_cl (int): number of Cl- ions to add (0 = minimal salt)
       script_only (bool): if True, only generate the tleap script
                            and do not run tleap

    Returns:
         (FileHandle, FileHandle, str): prmtop file, inpcrd file,
                                        tleap log text
        or:
            str: the tleap script (if script_only=True)

    '''
    _check_available('tleap')
    if len(amberpdbs) == 1:
        inputs = ['script', 'system.pdb']
    else:
        inputs = ['script']
        for i in range(len(amberpdbs)):
            inputs.append(f'component{i}.pdb')

    outputs = ['system.prmtop', 'system.inpcrd', 'STDOUT']
    script = "".join([f'source leaprc.{f}\n' for f in ff])

    if solvate:
        if solvate not in ['oct', 'box', 'cube']:
            raise ValueError(f'Error: unrecognised solvate option "{solvate}"')
        water_box = 'TIP3PBOX'
        for f in ff:
            if 'water.opc' in f:
                water_box = 'OPCBOX'
            elif 'water.tip4pew' in f:
                water_box = 'TIP4PEWBOX'
            elif 'water.tip4p2005' in f:
                water_box = 'TIP4P2005BOX'
            elif 'water.spce' in f:
                water_box = 'SPCEBOX'
    if het_names:
        if len(het_names) > 0:
            for r in het_names:
                _check_exists(Path(het_dir) / f'{r}.frcmod')
                _check_exists(Path(het_dir) / f'{r}.mol2')
                script += f'loadamberparams {r}.frcmod\n'
                script += f'{r} = loadmol2 {r}.mol2\n'
                inputs += [f'{r}.mol2', f'{r}.frcmod']

    if script_only:
        if len(amberpdbs) == 1:
            script += f"system = loadpdb {amberpdbs[0]}\n"
        else:
            for i, inpdb in enumerate(amberpdbs):
                script += f'component{i} = loadpdb {inpdb}\n'
            script += 'system = combine {'
            for i in range(len(amberpdbs)):
                script += f'component{i} '
            script += '}\n'

    else:
        if len(amberpdbs) == 1:
            script += "system = loadpdb system.pdb\n"
        else:
            for i, inpdb in enumerate(amberpdbs):
                script += f'component{i} = loadpdb component{i}.pdb\n'
            script += 'system = combine {'
            for i in range(len(amberpdbs)):
                script += f'component{i} '
            script += '}\n'

    if solvate == "oct":
        script += f"solvateoct system {water_box} {padding}\n"
    elif solvate == "cube":
        script += f"solvatebox system {water_box} {padding} iso\n"
    elif solvate == "box":
        script += f"solvatebox system {water_box} {padding}\n"
    if solvate is not None:
        script += f"addions system Na+ {n_na}\naddions system Cl- {n_cl}\n"

    script += "saveamberparm system system.prmtop system.inpcrd\nquit"
    if script_only:
        return script

    tleap = SubprocessTask('tleap -f script')
    tleap.set_inputs(inputs)
    tleap.set_outputs(outputs)
    fh = FileHandler()
    scriptfile = fh.create('scriptfile')
    scriptfile.write_text(script)
    args = [scriptfile] + amberpdbs
    if het_names:
        if len(het_names) > 0:
            for r in het_names:
                args += [f'{Path(het_dir)}/{r}.mol2',
                         f'{Path(het_dir)}/{r}.frcmod']
    prmtop, inpcrd, stdout = tleap(*args)
    if prmtop is None or inpcrd is None:
        raise RuntimeError(f'Error in leap: {stdout}')
    return prmtop, inpcrd, stdout


def ambpdb(inpcrd, prmtop):
    '''
    Convert an Amber inpcrd file to a PDB file.

    Args:
        inpcrd (str): input inpcrd file name
        prmtop (str): input prmtop file name
    '''
    _check_available('ambpdb')

    _ambpdb = SubprocessTask('ambpdb -p x.prmtop -c x.inpcrd > x.pdb')
    _ambpdb.set_inputs(['x.inpcrd', 'x.prmtop'])
    _ambpdb.set_outputs(['x.pdb'])
    outpdb = _ambpdb(inpcrd, prmtop)

    return outpdb


def indices_to_mask(indices):
    '''
    Convert a list of atom indices into an Amber-style atom mask
    '''
    mask = f'@{indices[0]}'
    if len(indices) == 1:
        return mask

    c = ' '
    for i in range(1, len(indices)):
        if indices[i] - indices[i-1] == 1:
            c += 'x'
        else:
            c += ' '

    for i in range(1, len(c)-1):
        if c[i] == ' ':
            if c[i-1] == 'x':
                mask += f'-{indices[i-1]}'
            mask += f',{indices[i]}'
    if c[-1] == 'x':
        mask += f'-{indices[-1]}'
    else:
        mask += f',{indices[-1]}'
    return mask
