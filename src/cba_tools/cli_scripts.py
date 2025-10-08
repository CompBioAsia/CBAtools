from argparse import ArgumentParser
import mdtraj as mdt
from cba_tools._version import __version__
from cba_tools.cba_tools import make_leap
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
    parser.add_argument('--no_opt', default=False, action='store_true',
                        help='Skip optimization at QM stage')
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('--version', action='version', version=__version__)

    parsed_args = parser.parse_args()
    parameterize(parsed_args.inpdb, parsed_args.het_name,
                 charge=parsed_args.het_charge,
                 gaff=parsed_args.forcefield, het_dir=parsed_args.het_dir,
                 overwrite=parsed_args.overwrite, no_opt=parsed_args.no_opt)


def make_leap_cli():
    parser = ArgumentParser(description="Generate tleap input script from PDB")
    parser.add_argument('--inpdb', help='Input PDB file', required=True)
    parser.add_argument('--outinpcrd', help='Output AMBER .inpcrd file',
                        required=True)
    parser.add_argument('--outprmtop', help='Output AMBER .prmtop file',
                        required=True)
    parser.add_argument('--forcefields', nargs='*', help='Force fields to use')
    parser.add_argument('--het_names', nargs='*',
                        help='Names of heterogen residues')
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

    result = make_leap(**vars(parsed_args))
    print(result)


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
    parser.add_argument("-u", "--uniprot_ids", nargs='*', required=False,
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
    parser.add_argument("-c", "--chains", nargs='*', required=False,
                        help="List of chain IDs to include in the output.")
    parser.add_argument("-u", "--uniprot_ids", nargs='*', required=False,
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
                             chains=args.chains, trim=not args.no_trim)
    out_pdb.save(args.outpdb)
    if args.log:
        with open(args.log, 'w') as log_file:
            log_file.write(log)
    else:
        print(log)


def sp_search_cli():
    parser = ArgumentParser(
        description="Search SwissProt for Uniprot codes"
                    " matching each chain in a protein structure."
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
            print(f'Matches for chain {i} ({t.topology.chain(i).chain_id}):')
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


def mesmy_cli():
    parser = ArgumentParser(
        description="Create a script for a multi-step Amber MD"
        " relaxation/equilibration workflow."
    )
    parser.add_argument("-i", "--inpcrd",
                        help="Input Amber CRD file.", required=True)
    parser.add_argument("-p", "--prmtop",
                        help="Input Amber PRMTOP file.", required=True)
    parser.add_argument("--version", action="version", version=__version__)

    args = parser.parse_args()

    t = mdt.load(args.inpcrd, top=args.prmtop)
    t_solute = t.atom_slice(
        t.topology.select('not water'))
    nres = t_solute.n_residues
    script = f"""#!/bin/bash

# An equilibration workflow.
# Designed for "standard" protein or protein/ligand systems
# in explicit solvent. Not optimised for membrane protein systems.
#
# Developed from scripts provided by the Hughes lab at the University
# of Montana.
#

# You may wish to modify some of the parameters below.
prmtop_file="{args.prmtop}"
inpcrd_file="{args.inpcrd}"
solute={nres} # number of residues in solute
T="310.0" # target temperature in K
PMEMD="pmemd.cuda" # name of your pmemd executable (e.g. may be "pmemd.cuda")

### DO NOT MODIFY BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING ###
# 1K step energy minimization with strong restraints on heavy atoms, no shake
cat > step1.in <<EOF
Min with strong restraints on heavy atoms, no shake
&cntrl
  imin = 1, ncyc = 50, maxcyc = 1000,
  ntpr = 50,
  ntr = 1, restraintmask = ':1-$solute & !@H=', restraint_wt = 5.0,
&end
EOF

# NVT MD with strong restraints on heavy atoms, shake, dt=.001, 15 ps
cat > step2.in <<EOF
NVT MD with strong restraints on heavy atoms, shake dt 0.001 15 ps
&cntrl
  imin = 0, nstlim = 30000, dt=0.001,
  ntpr = 50, ntwr = 500,
  iwrap = 1,
  ntc = 2, ntf = 2,
  ntt = 3, gamma_ln = 5, temp0 = $T, tempi = $T,
  ntr = 1, restraintmask = ':1-$solute & !@H=', restraint_wt = 5.0,
&end
EOF

# Energy minimization with relaxed restraints on heavy atoms, no shake
cat > step3.in <<EOF
Min with relaxed restraints on heavy atoms
&cntrl
  imin = 1, ncyc = 50, maxcyc = 1000,
  ntpr = 50,
  ntr = 1, restraintmask = ':1-$solute & !@H=', restraint_wt = 2.0,
&end
EOF

# Energy minimization with minimal restraints on heavy atoms, no shake
cat > step4.in <<EOF
Min with minimal restraints on heavy atoms
&cntrl
  imin = 1, ncyc = 50, maxcyc = 1000,
  ntpr = 50,
  ntr = 1, restraintmask = ':1-$solute & !@H=', restraint_wt = 0.1,
&end
EOF

# Energy minimization with no restraints, no shake
cat > step5.in <<EOF
Min with no restraints
&cntrl
  imin = 1, ncyc = 50, maxcyc = 1000,
  ntpr = 50,
&end
EOF

# NPT MD with shake and low restraints on heavy atoms, 20 ps dt=.002
cat > step6.in <<EOF
NPT MD with shake and low restraints on heavy atoms, 20 ps dt=.002
&cntrl
  imin = 0, nstlim = 10000, dt = 0.002,
  ntwx = 1000, ntpr = 50, ntwr = 500,
  iwrap = 1,
  ntc = 2, ntf = 2, ntb = 2,
  ntt = 3, gamma_ln = 5, temp0 = $T, tempi = $T,
  ntp = 1, taup = 1.0,
  ntr = 1, restraintmask = ':1-$solute & !@H=', restraint_wt = 1.0,
&end
EOF

# NPT MD with shake and minimal restraints on heavy atoms
cat > step7.in <<EOF
NPT MD with minimal restraints on heavy atoms, 20 ps dt=.002
&cntrl
  imin = 0, nstlim = 10000, dt=0.002,
  ntx = 5, irest = 1,
  ntwx = 1000, ntpr = 50, ntwr = 500,
  iwrap = 1,
  ntc = 2, ntf = 2, ntb = 2,
  ntt = 3, gamma_ln = 5, temp0 = $T, tempi = $T,
  ntp = 1, taup = 1.0,
  ntr = 1, restraintmask = ':1-$solute & !@H=', restraint_wt = 0.5,
&end
EOF

# NPT MD with shake and minimal restraints on backbone atoms, dt=0.002, 20 ps
cat > step8.in <<EOF
NPT MD with minimal restraints on backbone atoms, 20 ps dt=.002
&cntrl
  imin = 0, nstlim = 10000, dt=0.002,
  ntx = 5, irest = 1,
  ntwx = 1000, ntpr = 50, ntwr = 500,
  iwrap = 1,
  ntc = 2, ntf = 2, ntb = 2,
  ntt = 3, gamma_ln = 5, temp0 = $T, tempi = $T,
  ntp = 1, taup = 1.0,
  ntr = 1, restraintmask = ":1-$solute@H,N,CA,HA,C,O", restraint_wt = 0.5,
&end
EOF

# NPT MD with shake and no restraints, dt=0.002, 200 ps
cat > step9.in <<EOF
NPT MD no restraints, dt=0.002, 200 ps
&cntrl
  imin = 0, nstlim = 10000, dt=0.002,
  ntx = 5, irest = 1,
  ntwx = 1000, ntpr = 50, ntwr = 500,
  iwrap = 1,
  ntc = 2, ntf = 2, ntb = 2,
  ntt = 3, gamma_ln = 5, temp0 = $T, tempi = $T,
  ntp = 1, taup = 1.0,
&end
EOF

START="`date +%s.%N`"

# Minimization Phase - reference coords are updated each run
for RUN in step1 step2 step3 step4 step5 ; do
 echo "------------------------"
 echo "Minimization phase: $RUN"
 echo "------------------------"
 if [[ ! -f $RUN.out ]]; then
     echo "File -- $RUN.out -- does not exists. Running job..."
     runme=1
 else
     grep -q 'Total wall time' $RUN.out
     retval=$?
     if [ $retval -ne 0 ]; then
     echo "$RUN did not complete last time. Trying again..."
     runme=1
     else
         echo "$RUN has already completed.  Checking the next step."
         runme=0
     fi
 fi

 if [ $runme -eq 1 ]; then
     $PMEMD -O -i $RUN.in -p $prmtop_file -c $inpcrd_file \
        -ref $inpcrd_file -o $RUN.out -x $RUN.nc -r $RUN.ncrst \
            -inf $RUN.mdinfo
     grep -q 'Total wall time' $RUN.out
     retval=$?
     if [ $retval -ne 0 ]; then
         echo "Error - job failed at this step."
     exit $retval
     fi
 fi

 echo ""
 inpcrd_file="$RUN.ncrst"
done

# Equilibration phase - reference coords are last coords from minimize phase
REF=$inpcrd_file
for RUN in step6 step7 step8 step9 ; do

 echo "------------------------"
 echo "Equilibration phase: $RUN"
 echo "------------------------"
 if [[ ! -f $RUN.out ]]; then
     echo "File -- $RUN.out -- does not exists. Running job..."
     runme=1
 else
     grep -q 'Total wall time' $RUN.out
     retval=$?
     if [ $retval -ne 0 ]; then
     echo "$RUN did not complete last time. Trying again..."
     runme=1
     else
         echo "$RUN has already completed.  Checking the next step."
         runme=0
     fi
 fi

 if [ $runme -eq 1 ]; then
     $PMEMD -O -i $RUN.in -p $prmtop_file -c $inpcrd_file\
        -ref $REF -o $RUN.out -x $RUN.nc -r $RUN.ncrst -inf $RUN.mdinfo
     grep -q 'Total wall time' $RUN.out
     retval=$?
     if [ $retval -ne 0 ]; then
         echo "Error - job failed at this step."
     exit $retval
     fi
 fi

 echo ""
 inpcrd_file="$RUN.ncrst"
done

# Reset the time in the restart file to zero:
sed -i 's/0.3000000E+02/0.0000000E+00/g' step9.ncrst

STOP="`date +%s.%N`"
# The divide by 1 in the line below is to overcome a bug in bc...
TIMING=`echo "scale=1; ($STOP - $START) / 1;" | bc`
echo "Total run time: $TIMING seconds."
echo ""

exit 0
"""
    print(script)
