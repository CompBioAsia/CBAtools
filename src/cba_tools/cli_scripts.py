#!/usr/bin/env python3

from argparse import ArgumentParser
import mdtraj as mdt
from cba_tools.cba_tools import loopfix, param, make_refc


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
    parser.add_argument('--buffer',
                        help='minimum distance of solute atoms from box edge',
                        type=float, default=10.0)
    parser.add_argument('--het_dir', help='Directory for heterogen files')

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
    args = parser.parse_args()

    if not args.input_file or not args.donor_file or not args.output_file:
        parser.print_help()
        return

    t_in = mdt.load_pdb(args.input_file, standard_names=False)
    t_donor = mdt.load_pdb(args.donor_file, standard_names=False)
    fixed, log = loopfix(t_in,
                         t_donor,
                         trim=args.trim)

    fixed.save(args.output_file)
    print(log)


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
