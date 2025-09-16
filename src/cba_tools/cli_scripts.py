from argparse import ArgumentParser
import mdtraj as mdt
from cba_tools._version import __version__
from cba_tools.cba_tools import param
from cba_tools.cba_tools import sp_search, parameterize, alpha_check
from cba_tools.cba_tools import complete, rest_min, alpha_fix, alpha_get


def het_param_cli():
    parser = ArgumentParser(description="Parameterize a heterogen")
    parser.add_argument('--inpdb', help='Input PDB file', required=True)
    parser.add_argument('--het_name', help='Names of heterogen residues',
                        required=True)
    parser.add_argument('--het_charge', type=int, default=0,
                        help='Formal charge of heterogen')
    parser.add_argument('--forcefield', help='Force field to use',
                        default='gaff', choices=['gaff', 'gaff2'])
    parser.add_argument('--het_dir', help='Directory for heterogen files',
                        default='.')
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('--version', action='version', version=__version__)

    parsed_args = parser.parse_args()
    parameterize(parsed_args.inpdb, parsed_args.het_name,
                 charge=parsed_args.het_charge,
                 gaff=parsed_args.forcefield, het_dir=parsed_args.het_dir,
                 overwrite=parsed_args.overwrite)


def param_cli():
    parser = ArgumentParser(description="Generate AMBER input files from PDB")
    parser.add_argument('--inpdb', help='Input PDB file', required=True)
    parser.add_argument('--outinpcrd', help='Output AMBER .inpcrd file',
                        required=True)
    parser.add_argument('--outprmtop', help='Output AMBER .prmtop file',
                        required=True)
    parser.add_argument('--forcefields', nargs='*', help='Force fields to use')
    parser.add_argument('--het_names', nargs='*',
                        help='Names of heterogen residues')
    parser.add_argument('--het_charges', nargs='*',
                        help='Ligand formal charges')
    parser.add_argument('--solvate', help='Type of water box to use',
                        choices=['box', 'cube', 'oct'])
    parser.add_argument('--padding',
                        help='minimum distance of solute atoms from box edge',
                        type=float, default=10.0)
    parser.add_argument('--het_dir', help='Directory for heterogen files',
                        default='.')
    parser.add_argument('--ion_molarity', type=float,
                        help='Target ionic strength (M)')
    parser.add_argument('--version', action='version', version=__version__)

    parsed_args = parser.parse_args()
    # If param expects positional arguments, pass them directly
    result = param(**vars(parsed_args))
    return result


def prepare_protein_cli():

    parser = ArgumentParser(
        description="Prepare a PDB file for AMBER simulation."
    )
    parser.add_argument('-i', '--inpdb', help='Input PDB file', required=True)
    parser.add_argument('-o', '--outpdb', help='Output PDB file',
                        required=True)

    parsed_args = parser.parse_args()
    pdb_out = complete(parsed_args.inpdb)
    pdb_out.save(parsed_args.outpdb)


def rest_min_cli():
    parser = ArgumentParser(
        description="Perform restrained minimization on a protein PDB file."
    )
    parser.add_argument('-i', '--inpdb', help='Input PDB file', required=True)
    parser.add_argument('-o', '--outpdb', help='Output PDB file',
                        required=True)
    parser.add_argument('-r', '--refpdb', help='Reference PDB file')
    parser.add_argument('-l', '--logfile',
                        help='Log file for minimization output')

    parser.add_argument('--maxcyc', type=int, default=200,
                        help='Maximum number of minimization cycles')
    parser.add_argument('--kr', type=float, default=1.0,
                        help='Restraint force constant')
    parser.add_argument('--version', action='version', version=__version__)

    parsed_args = parser.parse_args()

    try:
        pdb_out, log = rest_min(parsed_args.inpdb,
                                pdbref=parsed_args.refpdb,
                                maxcyc=parsed_args.maxcyc,
                                kr=parsed_args.kr)
    except Exception as e:
        print("Error during minimization:", e)
        return
    pdb_out.save(parsed_args.outpdb)
    if parsed_args.logfile:
        with open(parsed_args.logfile, 'w') as log_file:
            log_file.write(log)


def alpha_check_cli():
    parser = ArgumentParser(
        description="Check how well a PDB file matches its UniProt sequences.")
    parser.add_argument("-i", "--inpdb",
                        help="Input PDB file.", required=True)
    parser.add_argument("-u", "--uniprot_ids", nargs='*', required=True,
                        help="List of UniProt IDs for the input structure.")
    parser.add_argument("-l", "--log", help="Log file for output.")
    parser.add_argument("--version", action="version", version=__version__)

    args = parser.parse_args()
    if not args.inpdb:
        parser.print_help()
        return
    log = alpha_check(args.inpdb, args.uniprot_ids)
    if args.log:
        with open(args.log, 'w') as log_file:
            log_file.write(log)
    else:
        print(log)


def alpha_get_cli():
    parser = ArgumentParser(
        description="Get Alphafold model for a UniProt IDs.")
    parser.add_argument("-u", "--uniprot_id", required=True,
                        help="UniProt ID to fetch.")
    parser.add_argument("-p", "--pdb", required=True,
                        help="PDB file to save the model.")
    parser.add_argument("--version", action="version", version=__version__)

    args = parser.parse_args()
    if not args.uniprot_id or not args.pdb:
        parser.print_help()
        return
    pdb = alpha_get(args.uniprot_id)
    pdb.save(args.pdb)


def alpha_fix_cli():
    parser = ArgumentParser(
        description="Fix missing residues in a PDB file using Alphafold.")
    parser.add_argument("-i", "--inpdb",
                        help="Input PDB file.", required=True)
    parser.add_argument("-o", "--outpdb",
                        help="Fixed PDB file.", required=True)

    parser.add_argument("-u", "--uniprot_ids", nargs='*', required=True,
                        help="List of UniProt IDs for the input structure.")
    parser.add_argument("-l", "--log", help="Log file for Alphafold output.")
    parser.add_argument("-n", "--no_trim", action="store_true",
                        help="Don't trim the fixed PDB file"
                        " to match the input.")
    parser.add_argument("--version", action="version", version=__version__)

    args = parser.parse_args()
    if not args.inpdb or not args.outpdb:
        parser.print_help()
        return
    out_pdb, log = alpha_fix(args.inpdb, args.uniprot_ids,
                             trim=not args.no_trim)
    out_pdb.save(args.outpdb)
    if args.log:
        with open(args.log, 'w') as log_file:
            log_file.write(log)
    else:
        print(log)


def sp_search_cli():
    parser = ArgumentParser(
        description="Search SwissProt for Uniprot codes"
                    " matching a protein structure."
    )
    parser.add_argument("-i", "--inpdb",
                        help="Input protein structure file.", required=True)
    parser.add_argument("-m", "--max_hits", type=int, default=1,
                        help="Maximum number of hits to return per chain.")
    parser.add_argument("--version", action="version", version=__version__)

    args = parser.parse_args()
    t = mdt.load(args.inpdb)
    seqs = t.topology.to_fasta()
    indent = ''
    for i, seq in enumerate(seqs):
        if len(seqs) > 1:
            print(f'Matches for chain {i}:')
            indent = '  '
        if len(seq) == 0:
            print(f"{indent}Skipping non-protein chain.")
            continue
        if len(seq) < 10:
            print(f"{indent}Skipping short ({len(seq)} residue chain.")
            continue
        result = sp_search(seq)
        for match in result[:args.max_hits]:
            uid = match['uniprotAccession']
            pid = float(match['percent_identity'])
            print(f"{indent}{uid} {pid:3.1f} %")
