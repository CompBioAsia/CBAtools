# A set of utility tools for the CompBioAsia Molecular
# Dynamics tutorials.
#
# These Python functions provide an easy interface to
# a range of third party tools that are useful for the
# preparartion of molecular systems for MD simulation.
# These include:
#
#   RDKit
#   OpenBabel
#   Chimera/ChimeraX
#   AmberTools
#   MDTraj
#
# The functions are:
#
#  smiles_to_traj: Generates a mdtraj.Trajectory for a
#                  molecule from a SMILES string. Tries
#                  to be intelligent about protonation
#                  states of any ionizable groups. Useful
#                  for ligand preparation. Internally uses
#                  RDKit and OpenBabel.
#
#  add_h:          Adds hydrogen atoms to heavy-atom only
#                  PDB format files. This is hard to get
#                  right every time with an automated tool,
#                  but the version here uses Chimera (or
#                  ChimeraX) which is often succesful. It
#                  also uses tools from AmberTools to 'clean
#                  up' the resulting structure, particularly
#                  setting the names of HIS residues to HID,
#                  HIE, or HIP depending on the predicted
#                  tautomer/ionization state. Internally uses
#                  Chimera(X) and pdb4amber.
#
#   param:         A complete AMBER-focussed workflow to
#                  prepare input files (coordinates and
#                  parameters) for MD simulation, from
#                  Complete PDB format files of the solute
#                  components (e.g. all-atom models of
#                  protein plus ligand). Includes automatic
#                  parameterization of non-standard
#                  residues (using gaff or gaff2), and addition
#                  of water boxes and neutralizing counterions.
#                  The tool only works for non-covalent ligands
#                  (no bonds between the ligand and the protein).
#                  Internally uses antechamber, parmchk2, and
#                  tleap.
#
#   loopfix:       A tool to fix missing residues in a PDB file.
#                  A user-supplied 'donor' PDB file is used to supply,
#                  where possible, loop residues missing from the 'acceptor'
#                  PDB file.

#   alpha_loopfix: A tool to fix missing residues in a PDB file
#                  It uses ChimeraX to search the Alphafold database
#                  for a suitable 'donor' structure, then uses it to
#                  fill in the missing loop residues.
#
#   complete:      A tool to complete a PDB file by adding missing
#                  atoms (heavy and hydrogen) using pdb4amber.
#
#   match_align:   A tool to superimpose two PDB files based on
#                  their sequence and structure. Similar to the
#                  Chimera(X) command of the same name.
#
#   make_refc:     A tool to generate Amber ".refc" files for restrained
#                  molecular dynamics simulations.
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

from crossflow.tasks import SubprocessTask
from crossflow.filehandling import FileHandler, FileHandle
from functools import cache
from enum import IntEnum
import shutil

import requests
from requests import exceptions
# from conditional_cache import lru_cache
from tempfile import NamedTemporaryFile
# from retry_requests import retry
from pathlib import Path
from time import sleep

#  Part 1: Various utilities


def smiles_to_traj(smi, pH=7.0):
    '''
    Convert an input SMILES representation to a (1 frame) MDTraj trajectory.

    Args:
        smi (str): Input SMILES string
        pH (float): target pH

    Returns:
        t (MDTrajectory): 3D structure for the molecule
        charge (int): Formal charge on the molecule.

    '''
    _check_available('obabel')
    smi_ph = SubprocessTask('echo {smi} | obabel -ismi -osmi -pH {pH}')
    smi_ph.set_inputs(['smi', 'pH'])
    smi_ph.set_outputs(['STDOUT'])
    smi_pH = smi_ph(smi, pH)

    charge = smi_pH.count('+]') - smi_pH.count('-]')
    mol_pH = Chem.MolFromSmiles(smi_pH)
    mol_pH_H = Chem.AddHs(mol_pH)
    rdDistGeom.EmbedMolecule(mol_pH_H)
    fh = FileHandler()
    tmp_pdb = fh.create('tmp.pdb')
    tmp_pdb.write_text(Chem.MolToPDBBlock(mol_pH_H))
    t_out = mdt.load_pdb(tmp_pdb, standard_names=False)
    return t_out, charge


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
    ''' Calculate the number of matches and gaps in a pairwise alignment.'''
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


def trim_sequence(t_in, t_ref):
    '''
    Trim terminii of t_in to match tref by sequence
    '''
    aln = smith_waterman(t_in.topology.to_fasta()[0],
                         t_ref.topology.to_fasta()[0])

    nt = 0
    while aln[1][nt] == '-':
        nt += 1
    nc = len(aln[1]) - 1
    while aln[1][nc] == '-':
        nc -= 1

    t_out = t_in.atom_slice(t_in.topology.select(f'resid {nt} to {nc}'))
    return t_out


def match_align(t_in, t_ref, cutoff=0.02):
    '''
    Superimpose t_in onto t_ref, based on sequence alignment

    The C-alpha atoms used for least-squares fitting are iteratively pruned
    until all pairs are within cutoff nanometers.

    '''
    alignment = smith_waterman(t_in.topology.to_fasta()[0],
                               t_ref.topology.to_fasta()[0])
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
    unconverged = True
    t_copy = mdt.Trajectory(t_in.xyz.copy(), t_in.topology.copy())
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
    return t_out, alignment


def _append(traj1, traj2, i):
    '''
    Append residue i from traj2 to traj1
    '''
    if traj1 is None:
        traj1 = traj2.atom_slice(
            traj2.topology.select(f'resid {i}'))
    else:
        traj1 = traj1.stack(traj2.atom_slice(
            traj2.topology.select(f'resid {i}')))
    return traj1


def _caca(traj1, i, traj2, j):
    '''
    Calculate the Calpha-Calpha distance between two residues
    '''
    indx1 = traj1.topology.select(f'resid {i} and name CA')
    indx2 = traj2.topology.select(f'resid {j} and name CA')
    if len(indx1) == 0 or len(indx2) == 0:
        raise ValueError(f'Error: no CA atoms found for {i} and {j}')
    dxyz = traj1.xyz[0, indx1] - traj2.xyz[0, indx2]
    return np.linalg.norm(dxyz, axis=1)


def merge(acceptor, donor, alignment, shoulder_width=3,
          min_ca_displacement=0.1, trim=True):
    '''
    Merge two MDTraj trajectories based on a sequence alignment.

    Missing parts of the acceptor trajectory are filled in with
    parts from the donor trajectory.

    It is assumed that the two trajectories are already structurally
    aligned.

    '''
    log = ''
    matches, mismatches, gaps = aln_score(alignment)
    if gaps == 0:
        print('Warning: no gaps detected')
        return acceptor
    aln2, aln1 = alignment

    nr = len(aln1)
    gapinfo = [a for a in aln1]
    if trim:
        i = 0
        while gapinfo[i] == '-':
            gapinfo[i] = 'x'
            i += 1
        i = len(gapinfo) - 1
        while gapinfo[i] == '-':
            gapinfo[i] = 'x'
            i -= 1

    for w in range(1, shoulder_width+1):
        for i in range(w, nr-w):
            if gapinfo[i-w] == '-' or gapinfo[i+w] == '-':
                gapinfo[i] = '?'
    # gapinfo holds info about shoulder residues and any trimming to be done.

    traj = None
    j1 = 0
    j2 = 0
    for i, r in enumerate(aln1):
        if r == '-':  # use t2
            if gapinfo[i] != 'x':
                log += f'inserting missing residue {donor.topology.residue(j2)}\n'
                traj = _append(traj, donor, j2)
            j2 += 1
        else:
            if gapinfo[i] == '?':
                caca = _caca(acceptor, j1, donor, j2)
                if caca < min_ca_displacement:
                    # shoulder residue, little deviation, keep acceptor
                    traj = _append(traj, acceptor, j1)
                else:
                    # shoulder residue, use donor
                    log += f'substituting shoulder residue {donor.topology.residue(j2)}\n'
                    traj = _append(traj, donor, j2)
                j2 += 1
                j1 += 1
            else:
                traj = _append(traj, acceptor, j1)
                j1 += 1
                if aln2[i] != '-':
                    j2 += 1
    newtop = mdt.Topology()
    cid = newtop.add_chain()
    for r in traj.topology.residues:
        rid = newtop.add_residue(r.name, cid)
        for a in r.atoms:
            _ = newtop.add_atom(a.name, a.element, rid)
    newtop._bonds = traj.topology._bonds
    return mdt.Trajectory(traj.xyz, newtop), log


def loopfix(acceptor, donor, cutoff=0.02, shoulder_width=3,
            min_ca_displacement=0.1, trim=True):
    '''
    remediate acceptor using residues from donor
    '''
    donor_at_acceptor, alignment = match_align(donor, acceptor, cutoff=cutoff)
    fixed_acceptor, log = merge(acceptor, donor_at_acceptor,
                                (alignment[0], alignment[2]),
                                trim=trim, shoulder_width=shoulder_width,
                                min_ca_displacement=min_ca_displacement)
    return fixed_acceptor, log


def complete(t_in):
    '''
    Complete a structure by adding missing atoms using pdb4amber.
    Args:
        t_in (MDTrajectory): input structure

    '''

    _check_available('pdb4amber')
    pdb4amber = SubprocessTask('pdb4amber -i in.pdb --add-missing-atoms'
                               ' --reduce > out.pdb')
    pdb4amber.set_inputs(['in.pdb'])
    pdb4amber.set_outputs(['out.pdb'])
    out = pdb4amber(t_in)
    # CONECT records added by pdb4amber end up bogus...
    if 'CONECT' in out.read_text():
        text = '\n'.join([line for line in out.read_text().split('\n')
                         if 'CONECT' not in line]) + '\n'
        out.write_text(text)
    t_out = mdt.load_pdb(out, standard_names=False)
    return t_out


def h_strip(t_in):
    '''
    Strip all hydrogen atoms from a trajectory.

    Args:
        t_in (MDTrajectory): input trajectory

    Returns:
        mdt.Trajectory: the stripped trajectory

    '''
    if not isinstance(t_in, mdt.Trajectory):
        raise TypeError('Input must be an MDTraj trajectory')

    h_indices = t_in.topology.select('element H')
    t_out = t_in.atom_slice(np.setdiff1d(np.arange(t_in.n_atoms), h_indices))
    return t_out


def gapsplit(t_in):
    '''
    Remove bogus peptide bonds from a trajectory.

    MDTraj will bond consecutive amino acids even if there is
    a gap between them. This function removes such bonds.

    Args:
        trajin (mdt.Trajectory): input trajectory

    Returns:
        mdt.Trajectory: the cleaned trajectory

    '''
    if not isinstance(t_in, mdt.Trajectory):
        raise TypeError('Input must be an MDTraj trajectory')

    bonds = t_in.topology._bonds
    ibonds = [[b[0].index, b[1].index] for b in bonds]
    bls = mdt.compute_distances(t_in, ibonds)[0]

    good_bonds = []
    for i, bl in enumerate(bls):
        bond_id = bonds[i][0].name + bonds[i][1].name
        if bond_id == 'CN' or bond_id == 'NC':
            # This is a peptide bond, check if it is bogus
            if bl < 0.3:
                # This is an OK bond, keep it
                good_bonds.append(bonds[i])
        else:
            good_bonds.append(bonds[i])
    t_out = mdt.Trajectory(t_in.xyz, t_in.topology)
    t_out.topology._bonds = good_bonds
    return t_out


#  Part 3: Web service based tools

class Blaster():
    def __init__(self, session=None):
        if not session:
            #  self.session = retry(requests.Session(), retries=5,
            #                      backoff_factor=0.2)
            self.session = requests.Session()
        else:
            self.session = session
        self.url_base = 'https://blast.ncbi.nlm.nih.gov/Blast.cgi'

    def submit(self, fasta):
        '''
        Submit a Blast job to find the Uniprot IDs
        of the best matches to a sequence

        '''
        params = {
            'CMD': 'Put',
            'QUERY': fasta,
            'DATABASE': 'swissprot',
            'PROGRAM': 'blastp',
            'HITLIST_SIZE': 3,
            'ALIGNMENTS': 3
        }
        try:
            response = self.session.get(self.url_base, params=params)
        except exceptions.ConnectionError:
            raise exceptions.ConnectionError("Error: can't reach Blast server")

        content = response.text.split('\n')

        for c in content:
            if 'RID =' in c:
                rid = c.split()[2]

        content = response.text.split('\n')
        for c in content:
            if 'estimate' in c:
                delay = int(c.split()[8])
        return rid, delay

    def status(self, rid):
        '''
        Check the status of a job

        '''
        response = self.session.get(self.url_base,
                                    params={'CMD': 'Get', 'RID': rid})
        if 'Status=' in response.text:
            ioff = response.text.index('Status=')
            text = response.text[ioff+7:ioff+17].split()[0]
            return text
        else:
            return 'UNKNOWN'

    def retrieve(self, rid):
        '''
        Retrieve the results of a job

        '''
        response = self.session.get(self.url_base,
                                    params={'CMD': 'Get',
                                            'RID': rid,
                                            'FORMAT_TYPE': 'JSONSA'})

        data = response.json()
        alignments = data['Seq_annot']['data']['align']
        accessions = []
        for alignment in alignments:
            accessions.append(alignment['segs']['denseg']['ids'][1][
                'swissprot']['accession'])
        return accessions


def search_uniprot_ids(t):
    '''
    Search for Unicode accession Id(s) for the provided (protein) trajectory

    This approach is generally faster than trying blast

    '''
    # Chop the trajectory sequence into fragments
    t = gapsplit(t)
    ms = t.topology.find_molecules()
    frag_len = 20
    frag_min = 3

    frags = []
    for m in ms:
        indx = sorted([a.index for a in m])
        seq = t.topology.subset(indx).to_fasta()[0]
        frgs = [seq[i:i+frag_len] for i in range(0, len(seq), frag_len)]
        for f in frgs:
            if len(f) > frag_min:
                frags.append(f)

    # The Uniprot peptide search API:
    endpoint = 'https://peptidesearch.uniprot.org/asyncrest/'

    # 1. Submit:
    data = {'peps': frags,
            'lEQi': 'off',
            'spOnly': 'on'
            }
    response = requests.post(endpoint, data=data)
    if response.status_code == 202:
        loc_url = response.headers['Location']
    else:
        raise Exception('Request not accepted')

    # 2. Poll for completion:
    status_code = 303
    max_polls = 10
    i_poll = 1
    while status_code == 303 and i_poll < max_polls:
        sleep(5)
        response = requests.get(loc_url)
        status_code = response.status_code
        i_poll += 1

    # 3. Get results:
    if status_code == 303:
        raise Exception('Error: request timed out')
    elif status_code != 200:
        raise Exception(f'Error: {response}')
    uids = [u for u in response.text.split(',') if '-' not in u]
    return uids


#  @lru_cache(condition=lambda result: result is not None)
def search_accession(fasta, session=None):
    '''
    See if you can find a Uniprot accession Id for the sequence in traj

    This version uses the NCBI Blast service, which can be slow...
    '''
    blaster = Blaster(session=session)
    rid, delay = blaster.submit(fasta)
    waiting = True
    while waiting:
        print('waiting...')
        sleep(delay)
        status = blaster.status(rid)
        waiting = status == 'WAITING'

    try:
        accessions = blaster.retrieve(rid)
    except Exception as e:
        print(type(e))
        return
    return accessions


def alpha_get(uniprot_id, session=None):
    '''
    Get the Alphafold structure with the given Uniprot Id
    If it exists
    '''
    if not session:
        #  session = retry(requests.Session(), retries=5, backoff_factor=0.2)
        session = requests.Session()
    base_url = f'https://alphafold.com/api/prediction/{uniprot_id}'
    response = session.get(base_url)
    data = response.json()
    if 'error' in data:
        raise ValueError(f'Error: {uniprot_id} not in Alphafold database')

    pdb_url = data[0]['pdbUrl']
    response2 = session.get(pdb_url)
    with NamedTemporaryFile(suffix='.pdb') as f:
        f.write(response2.text.encode('utf-8'))
        t_out = mdt.load_pdb(f.name, standard_names=False)
    return t_out


def _check_available(cmd):
    '''
    Little utility to check a required command is available

    '''
    if shutil.which(cmd) is None:
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
            print('Warning, existing file {path} will be over-written')
        else:
            raise FileExistsError('Error: {path} already exists')


def add_h(t_in, chimera='chimera', mode='amber'):
    '''
    Add hydrogen atoms to a structure, using Chimera or ChimeraX

    Args:
        t_in (MDTrajectory): stucturemissing hydrogens
        chimera (str): command to invoke Chimera/ChimeraX
        mode (str): Adjust residue names for chosen software
                    (only 'amber' is currently supported).
    '''
    _check_available(chimera)
    chimera_version = SubprocessTask(f"{chimera} --version > version")
    chimera_version.set_outputs(["version"])
    version = chimera_version.run()
    if 'ChimeraX' in version.read_text():
        chimera_type = 'chimerax'
    else:
        chimera_type = 'chimera'

    fh = FileHandler()
    script = fh.create('script')
    if chimera_type == 'chimerax':
        script.write_text('open infile.pdb\naddh\nsave outfile.pdb #1\nquit')
    else:
        script.write_text('open infile.pdb\naddh\nwrite 0  outfile.pdb\nstop')
    addh = SubprocessTask(f"{chimera} --nogui < script")
    addh.set_inputs(['script', 'infile.pdb'])
    addh.set_outputs(['outfile.pdb'])
    addh.set_constant('script', script)

    outfile = addh(t_in)
    if mode == 'amber':
        _check_available('pdb4amber')
        pdb4amber = SubprocessTask('pdb4amber -i infile.pdb -o outfile.pdb')
        pdb4amber.set_inputs(['infile.pdb'])
        pdb4amber.set_outputs(['outfile.pdb'])
        amberpdb = pdb4amber(outfile)
        t_out = mdt.load_pdb(amberpdb, standard_names=False)
    else:
        t_out = mdt.load_pdb(outfile, standard_names=False)
    n_h_in = len(t_in.topology.select('mass < 2.0'))
    n_h_out = len(t_out.topology.select('mass < 2.0'))
    n_h_added = n_h_out - n_h_in
    log = f'fix added {n_h_added} hydrogen atoms'
    if mode == 'amber':
        for i, r in enumerate(t_in.topology.residues):
            r_new = t_out.topology.residue(i)
            if r.name != r_new.name:
                log += f'{r.name}{r.resSeq} is now {r_new.name}\n'
    return t_out, log


def alpha_match(prot_in, chimerax='chimerax', trim=True):
    '''
    Find an Alphafold structure that matches the supplied protein
    structure file.

    The input can be an MDTraj trajectory, a Crossflow FileHandle, a Path, or a
    filename (string)
    '''
    _check_available(chimerax)

    if not isinstance(prot_in, (str, Path, mdt.Trajectory, FileHandle)):
        raise TypeError(f'Unsupported input type {type(prot_in)})')

    if isinstance(prot_in, (str, Path)):
        _check_exists(prot_in)

    if not isinstance(prot_in, mdt.Trajectory):
        # Convert to MDTraj trajectory
        prot_in = mdt.load(prot_in)
    if prot_in.topology.n_chains != 1:
        raise ValueError('Input structure must contain exactly one chain')

    fh = FileHandler()
    script = fh.create('script')
    if trim:
        script.write_text('open infile.pdb\nalphafold match #1\n'
                          'save outfile.pdb #2\nquit')
    else:
        script.write_text('open infile.pdb\nalphafold match #1 trim false\n'
                          'save outfile.pdb #2\nquit')
    find_af = SubprocessTask(f"{chimerax} --nogui < script")
    find_af.set_inputs(['script', 'infile.pdb'])
    find_af.set_outputs(['outfile.pdb', 'STDOUT'])
    find_af.set_constant('script', script)

    outfile, log = find_af(prot_in)
    lines = log.split('\n')
    for i, l in enumerate(lines):
        if 'ERROR' in l:
            raise Exception(f'Error: {lines[i+1]}')

    return mdt.load_pdb(outfile, standard_names=False), log


def alpha_loopfix(inpdb, outpdb,
                  max_shoulder_size=3,
                  min_ca_displacement=0.1,
                  chimerax='chimerax',
                  trim=True):
    """
    Fix missing residues in a PDB file using AlphaFold.
    Args:
        inpdb (str): input PDB file name
        outpdb (str): output PDB file name
        max_shoulder_size (int): maximum size of a loop shoulder
                                 to be replaced by AlphaFold
        min_ca_displacement (float): minimum displacement of the CA
                                     atom from the original structure
                                     to be replaced by AlphaFold
        chimerax (str): command to invoke ChimeraX
        trim (bool): whether to trim the AlphaFold
                      structure to match the input structure

    """
    _check_exists(inpdb)
    _check_available(chimerax)
    if not trim:
        print('Warning: trim=False, the output structure will contain'
              ' all residues from the AlphaFold structure.')

    t = mdt.load_pdb(inpdb, standard_names=False)

    t_alpha, log = alpha_match(t, trim=False)
    lines = log.split('\n')
    for i, l in enumerate(lines):
        if 'sequence similarity' in l:
            print(l)
        elif 'WARNING' in l:
            print(f'{l} {lines[i+1]}')

    t_fixed, log = loopfix(t, t_alpha, trim=trim, shoulder_width=max_shoulder_size,
                           min_ca_displacement=min_ca_displacement)
    t_fixed.save(outpdb)
    print(log)
    print(f'Fixed structure saved as {outpdb}.')


#  Part 5: Tools for (Amber) MD simulation preparation


def param(inpdb, outprmtop, outinpcrd, het_names=None, het_charges=None,
          het_dir='.',
          overwrite=False,
          forcefields=None,
          solvate=None,
          buffer=10.0):
    """
    Parameterize a PDB file for AMBER simulations.

    Args:
        inpdb (str): name of input PDB file
        outprmtop (str): name of output prmtop file
        outinpcrd (str): name of output inpcrd file
        het_names (None or list): 3-letter residue names for heterogens
        het_charges (None or list): Formal charges of each heterogen
        forcefields (None or list): List of forcefields to use
        solvate (None or str): Solvation option - can be 'box',
                               'cube', or 'oct'.
        buffer (float): minimum distance from any solute atom
                        to a periodic box boundary (Angstroms)

    """
    _check_exists(inpdb)
    if not forcefields:
        print('Warning: no forcefields specified, '
              'defaulting to "protein.ff14SB"')
        forcefields = ['protein.ff14SB']
    if solvate:
        if solvate not in ['oct', 'box', 'cube']:
            raise ValueError(f'Error: unrecognised solvate option "{solvate}"')
        water_ff = False
        for ff in forcefields:
            if 'water' in ff:
                water_ff = True
        if not water_ff:
            print('Warning: no water forcefield specified but'
                  ' solvation required.')
            if 'protein.ff19SB' in forcefields:
                print('Defaulting to "water.opc" forcefield.')
                forcefields.append('water.opc')
            else:
                print('Defaulting to "water.tip3p" forcefield.')
                forcefields.append('water.tip3p')

    if het_names is not None:
        if 'gaff' not in forcefields and 'gaff2' not in forcefields:
            print('Warning - heterogens are present but no gaff/gaff2 '
                  'forcefield has been specified.')
            print('Will default to using "gaff".')
            forcefields.append('gaff')
            gaff = 'gaff'
        elif 'gaff' in forcefields:
            gaff = 'gaff'
        else:
            gaff = 'gaff2'

        for h_name, h_charge in zip(het_names, het_charges):
            print(f'parameterizing heterogen {h_name}')
            parameterize(inpdb, h_name, h_charge, het_dir=het_dir, gaff=gaff,
                         overwrite=overwrite)

    prmtop, inpcrd = leap(inpdb, forcefields, het_names=het_names,
                          solvate=solvate, buffer=buffer, het_dir=het_dir)
    prmtop.save(outprmtop)
    print(f'Parameters written to {outprmtop}')
    inpcrd.save(outinpcrd)
    print(f'Coordinates written to {outinpcrd}')


@cache
def parameterize(source, residue_name, charge=0, gaff='gaff',
                 het_dir='.', overwrite=False):
    '''
    Paramaterize a non-standard residue (heterogen)

    Uses antechamber and parmchk2 to generate .mol2 and .frcmod files.

    Args:
       source (str): the PDB file name
       residue_name (str): the three-letter residue code for the heterogen
       charge: the formal charge on the heterogen

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
        raise ValueError('Error: no residue {residue_name} found in {source}')

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
    if gaff == 'gaff':
        antechamber = SubprocessTask('antechamber -i infile.pdb -fi pdb'
                                     ' -o outfile.mol2 -fo mol2 -c bcc'
                                     ' -nc {charge}')
    else:
        antechamber = SubprocessTask('antechamber -i infile.pdb -fi pdb'
                                     ' -o outfile.mol2 -fo mol2 -c bcc'
                                     ' -nc {charge} -at gaff2')
    antechamber.set_inputs(['infile.pdb', 'charge'])
    antechamber.set_outputs(['outfile.mol2'])
    outmol2 = antechamber(traj_het, charge)
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


def leap(amberpdb, ff, het_names=None, solvate=None, buffer=10.0, het_dir='.'):
    '''
    Parameterize a molecular system using tleap.

    Args:
       amberpdb str): An Amber-compliant PDB file
       ff (list): The force fields to use.
       het_names (list): List of parameterised heterogens
       solvate (str or None): type of periodic box ('box', 'cube', or 'oct')
       buffer (float): Clearance between solute and any box edge (Angstroms)
       het_dir (str or Path): location of the directory containing heterogen parameters

    '''
    inputs = ['script', 'system.pdb']
    outputs = ['system.prmtop', 'system.inpcrd']
    script = "".join([f'source leaprc.{f}\n' for f in ff])

    if solvate:
        if solvate not in ['oct', 'box', 'cube']:
            raise ValueError(f'Error: unrecognised solvate option "{solvate}"')
    if het_names:
        if len(het_names) > 0:
            for r in het_names:
                _check_exists(Path(het_dir) / f'{r}.frcmod')
                _check_exists(Path(het_dir) / f'{r}.mol2')
                script += f'loadamberparams {r}.frcmod\n'
                script += f'{r} = loadmol2 {r}.mol2\n'
                inputs += [f'{r}.mol2', f'{r}.frcmod']

    script += "system = loadpdb system.pdb\n"
    if solvate == "oct":
        script += f"solvateoct system TIP3PBOX {buffer}\n"
    elif solvate == "cube":
        script += f"solvatebox system TIP3PBOX {buffer} iso\n"
    elif solvate == "box":
        script += f"solvatebox system TIP3PBOX {buffer}\n"
    if solvate is not None:
        script += "addions system Na+ 0\naddions system Cl- 0\n"
    script += "saveamberparm system system.prmtop system.inpcrd\nquit"

    tleap = SubprocessTask('tleap -f script')
    tleap.set_inputs(inputs)
    tleap.set_outputs(outputs)
    fh = FileHandler()
    scriptfile = fh.create('scriptfile')
    scriptfile.write_text(script)
    args = [scriptfile, amberpdb]
    if het_names:
        if len(het_names) > 0:
            for r in het_names:
                args += [f'{Path(het_dir)}/{r}.mol2', f'{Path(het_dir)}/{r}.frcmod']
    prmtop, inpcrd = tleap(*args)
    return prmtop, inpcrd


def _alpha_loopfix(inpdb, outpdb,
                   max_shoulder_size=3,
                   min_ca_displacement=0.1,
                   chimerax='chimerax',
                   trim=True):
    """
    Fix missing residues in a PDB file using AlphaFold.
    Args:
        inpdb (str): input PDB file name
        outpdb (str): output PDB file name
        max_shoulder_size (int): maximum size of a loop shoulder
                                 to be replaced by AlphaFold
        min_ca_displacement (float): minimum displacement of the CA
                                     atom from the original structure
                                     to be replaced by AlphaFold
        chimerax (str): command to invoke ChimeraX
        trim (bool): whether to trim the AlphaFold
                      structure to match the input structure

    """
    _check_exists(inpdb)
    _check_available(chimerax)
    if not trim:
        print('Warning: trim=False, the output structure will contain'
              ' all residues from the AlphaFold structure.')

    t = mdt.load_pdb(inpdb, standard_names=False)

    t = gapsplit(t)

    ms = t.topology.find_molecules()
    if len(ms) == 1:
        print('No gaps to fill.')
        if trim:
            return
    elif len(ms) == 2:
        print('Structure contains 1 gap.')
    else:
        print(f'Structure contains {len(ms)-1} gaps.')

    outfile, log = alpha_match(t, chimerax=chimerax, trim=trim)

    lines = log.split('\n')
    for i, l in enumerate(lines):
        if 'sequence similarity' in l:
            print(l)
        elif 'WARNING' in l:
            print(f'{l} {lines[i+1]}')

    if not outfile:
        raise RuntimeError(f'Error: AlphaFold search failed.\n{log}.')

    ta = mdt.load_pdb(outfile, standard_names=False)

    residues = []
    for i in range(ta.topology.n_residues):
        r = dict(alpha_res=ta.topology.residue(i), alpha_idx=i)
        residues.append(r)

    ref_seq = ta.topology.to_fasta()[0]
    new_indx = []
    d_from_gap = []
    for im, m in enumerate(ms):
        sel = [a.index for a in m]
        frag = t.topology.subset(sel)
        frag_seq = frag.to_fasta()[0]
        start = ref_seq.index(frag_seq)
        fl = len(frag_seq)
        for i in range(fl):
            new_indx.append(i + start)
            if im == 0:
                d_from_gap.append(fl - i)
            elif im == len(ms) - 1:
                d_from_gap.append(i+1)
            else:
                d_from_gap.append(min(i, fl-i))

    pair_d = []
    for i, j in enumerate(new_indx):
        sel_i = t.topology.select(f'resid {i} and name CA')[0]
        sel_j = ta.topology.select(f'resid {j} and name CA')[0]
        dxyz = t.xyz[0, sel_i] - ta.xyz[0, sel_j]
        pair_d.append(np.linalg.norm(dxyz))

    for i, j in enumerate(new_indx):
        residues[j]['xtal_idx'] = i
        residues[j]['d_from_gap'] = d_from_gap[i]
        residues[j]['xtal_res'] = t.topology.residue(i)
        residues[j]['pair_d'] = pair_d[i]

    if 'pair_d' not in residues[0]:
        # Gap (extra N-termnal residues) at the start, use the first residue f
        # rom the alphafold structure
        alist = [a.index for a in residues[0]['alpha_res'].atoms]
        new_traj = ta.atom_slice(alist)
    elif residues[0]['d_from_gap'] <= max_shoulder_size and \
      residues[0]['pair_d'] > min_ca_displacement:
        alist = [a.index for a in residues[0]['alpha_res'].atoms]
        new_traj = ta.atom_slice(alist)
    else:
        alist = [a.index for a in residues[0]['xtal_res'].atoms]
        new_traj = t.atom_slice(alist)
    for r in residues[1:]:
        if 'pair_d' not in r:
            # Use the alphafold structure for this residue
            alist = [a.index for a in r['alpha_res'].atoms]
            new_traj = new_traj.stack(ta.atom_slice(alist))
            print(f"Inserting missing residue {r['alpha_res']}.")
        elif (r['d_from_gap'] <= max_shoulder_size and
              r['pair_d'] > min_ca_displacement):
            # Use the alphafold structure for this residue
            alist = [a.index for a in r['alpha_res'].atoms]
            new_traj = new_traj.stack(ta.atom_slice(alist))
            print(f"Replacing {r['xtal_res']} with {r['alpha_res']}.")
        else:
            # Use the crystal structure for this residue
            alist = [a.index for a in r['xtal_res'].atoms]
            new_traj = new_traj.stack(t.atom_slice(alist))

    new_top = mdt.Topology()
    c = new_top.add_chain()
    for r in new_traj.topology.residues:
        nr = new_top.add_residue(r.name, c, r.resSeq)
        for a in r.atoms:
            new_top.add_atom(a.name, a.element, nr)
    new_traj.topology = new_top
    new_traj.save(outpdb)
    print(f'Fixed structure saved as {outpdb}.')


def make_refc(pdb, inpcrd, prmtop, refc):
    '''
    Make a reference coordinate file for an Amber simulation.

    Args:
        pdb (str): input PDB file name
        inpcrd (str): input inpcrd file name
        prmtop (str): input prmtop file name
        refc (str): output reference coordinate file name

    '''
    _check_exists(pdb)
    _check_exists(inpcrd)
    _check_exists(prmtop)

    tr = mdt.load_pdb(pdb, standard_names=True)
    ti = mdt.load(inpcrd, top=prmtop)
    n_ref = tr.topology.n_atoms
    n_inp = ti.topology.n_atoms
    if n_ref > n_inp:
        raise ValueError(f'Error: number of atoms in {pdb} ({n_ref}) '
                         f'exceeds the number of atoms in {inpcrd} ({n_inp})')
    tir = ti.atom_slice(np.arange(n_ref))
    # Check if the topologies match, at least by atom names:
    for i in range(tir.topology.n_atoms):
        if tir.topology.atom(i).name != tr.topology.atom(i).name:
            raise ValueError(f'Error: atom {i} names do not match '
                             f'({tir.topology.atom(i).name} in {inpcrd}, '
                             f'{tr.topology.atom(i).name} in {pdb})')
    # Superpose the input coordinates to the reference coordinates
    tr = tr.superpose(tir)

    ti.xyz[0, :n_ref] = tr.xyz[0, :n_ref]
    ti.save(refc)
    print(f'Reference coordinates saved as {refc}.')
