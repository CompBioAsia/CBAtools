from argparse import ArgumentParser
import mdtraj as mdt
from cba_tools.cba_tools import loopfix, param, make_refc, alpha_match, sp_search
from cba_tools.cba_tools import complete, rest_min, alpha_loopfix, alpha_fix, alpha_check


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

    parsed_args = parser.parse_args()
    # If param expects positional arguments, pass them directly
    result = param(**vars(parsed_args))
    return result


def loopfix_cli():
    parser = ArgumentParser(description="Fix loops in PDB files.")
    parser.add_argument("-i", "--input_file", help="Input PDB file.")
    parser.add_argument("-r", "--donor_file",
                        help="Donor PDB file for loop fixing.")
    parser.add_argument("-o", "--output_file", help="Fixed PDB file.")
    parser.add_argument("-t", "--trim", action="store_true",
                        help="Trim the fixed PDB file to match the input.")
    parser.add_argument("-w", "--shoulder_width", type=int, default=3,
                        help="Shoulder width for loop fixing.")
    args = parser.parse_args()

    if not args.input_file or not args.donor_file or not args.output_file:
        parser.print_help()
        return

    t_in = mdt.load_pdb(args.input_file, standard_names=False)
    t_donor = mdt.load_pdb(args.donor_file, standard_names=False)
    fixed, chunks = loopfix(
        t_in,
        t_donor,
        trim=args.trim,
        shoulder_width=args.shoulder_width)

    fixed.save(args.output_file)
    for chunk in chunks:
        msg = f"residues {chunk['start']} to {chunk['end']} "
        msg += f"built from {chunk['source']}"
        print(msg)
    print(f"Fixed structure saved as {args.output_file}.")


def make_refc_cli():
    parser = ArgumentParser(
        description="Generate reference coordinates for AMBER MD from PDB file"
    )
    parser.add_argument('--pdb', help='Input PDB file', required=True)
    parser.add_argument('--refc', help='Output reference coordinates file',
                        required=True)
    parser.add_argument('--inpcrd', help='Input AMBER .inpcrd file',
                        required=True)
    parser.add_argument('--prmtop', help='Input AMBER .prmtop file',
                        required=True)
    parsed_args = parser.parse_args()
    result = make_refc(**vars(parsed_args))
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
    parser.add_argument('-l', '--logfile', help='Log file for minimization output')

    parser.add_argument('--maxcyc', type=int, default=200,
                        help='Maximum number of minimization cycles')
    parser.add_argument('--kr', type=float, default=1.0,
                        help='Restraint force constant')
    
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
    parser.add_argument("-t", "--trim", action="store_true",
                        help="Trim the fixed PDB file to match the input.")

    args = parser.parse_args()
    if not args.inpdb or not args.outpdb:
        parser.print_help()
        return
    out_pdb, log = alpha_fix(args.inpdb, args.uniprot_ids, trim=args.trim)
    out_pdb.save(args.outpdb)
    if args.log:
        with open(args.log, 'w') as log_file:
            log_file.write(log)


def alpha_loopfix_cli():
    parser = ArgumentParser(
        description="Fix missing loops in a PDB file using Alphafold.")
    parser.add_argument("-i", "--input_file",
                        help="Input PDB file.", required=True)
    parser.add_argument("-o", "--output_file",
                        help="Fixed PDB file.", required=True)
    parser.add_argument("-t", "--trim", action="store_true",
                        help="Trim the fixed PDB file to match the input.")
    parser.add_argument("-u", "--uniprot_ids", nargs='*',
                        help="List of UniProt IDs for the input structure.")
    parser.add_argument("-w", "--shoulder_width", type=int, default=3,
                        help="Shoulder width for loop fixing.")
    args = parser.parse_args()

    alpha_loopfix(args.input_file, args.output_file, trim=args.trim,
                  uniprot_ids=args.uniprot_ids,
                  max_shoulder_size=args.shoulder_width)


def alpha_match_cli():
    parser = ArgumentParser(
        description="Find Alphafold structure for each chain"
                    " in the supplied protein structure."
    )
    parser.add_argument("-i", "--input_file",
                        help="Input protein structure file.", required=True)
    parser.add_argument("-m", "--max_matches", type=int, default=None,
                        help="Maximum number of matches to return per chain.")
    args = parser.parse_args()

    result = alpha_match(args.input_file, max_matches=args.max_matches)

    for chain, matches in result.items():
        print(f"Chain {chain}:")
        for match in matches:
            print(f" - {match['uniprotAccession']:10s}:"
                  f" {match['identity']} match")


def sp_search_cli():
    parser = ArgumentParser(
        description="Search for Uniprot matches to a protein structure."
    )
    parser.add_argument("-i", "--inpdb",
                        help="Input protein structure file.", required=True)
    
    args = parser.parse_args()
    t = mdt.load(args.inpdb)
    seqs = t.topology.to_fasta()
    indent = ''
    for i, seq in enumerate(seqs):
        if len(seqs) > 1:
            print(f'Matches for chain {i}:')
            indent = '  '
        if len(seq) < 10:
            print(f"{indent}Sequence too short ({len(seq)} residues), skipping.")
            continue
        result = sp_search(seq)
        for match in result:
            print(f"{indent}{match['uniprotAccession']} {float(match['percent_identity']):3.1f} %")
