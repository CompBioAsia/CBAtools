from pathlib import Path
from crossflow.filehandling import FileHandler

from rdkit import Chem
from rdkit.Chem import AllChem
import mdtraj as mdt

import numpy as np
import yaml
import sys
import logging

from cba_tools.utils import bcolors, suppress_tracebacks, installed
from cba_tools.tasks import pdb4amber, pdb2pqr, antechamber, parmchk2, tleap

# Suppress tracebacks
suppress_tracebacks()

# Tell the RDKit's C++ backend to log to use the python logger:
from rdkit import rdBase
rdBase.LogToPythonLogger()

def validate_yaml_input(yaml_file):
    '''
    Check the input and add defaults as required
    '''
    success = True

    f = Path(yaml_file)
    if not f.exists():
        print(f'  {bcolors.FAIL}Error:{bcolors.ENDC} cannot find input yaml file {yaml_file}')
        success = False
        return data, success
    
    yaml_data = f.read_text()
    data = yaml.safe_load(yaml_data)
    required_keys = ['system_name', 'source']
    for key in required_keys:
        if key not in data:
            print(f'  {bcolors.FAIL}Error - missing {key} parameter'
                             f' in input yaml file.{bcolors.ENDC}')
            success = False

    defaults = {'non_standard_residues': [],
                'protein_ff': 'ff14SB',
                'ligand_ff': 'gaff',
                'target_pH': 7.4,
                'water_model': 'TIP3P',
                'target_ion_molarity': 0.15, 
                'salt': 'NaCl',
                'pbc': 'cubic',
                'padding': 12.0,
                }

    for key in defaults:
        if key not in data:
            data[key] = defaults[key]
            print(f'  {bcolors.WARNING}Warning:{bcolors.ENDC} No value for parameter {bcolors.BOLD}{key}{bcolors.ENDC}'
                  f' provided,'
                  f' setting to default of {bcolors.BOLD}{defaults[key]}'
                  f'{bcolors.ENDC}.')

    valid_protein_force_fields = ['ff14SB', 'ff19SB']
    valid_ligand_force_fields = ['gaff', 'gaff2']
    valid_ion_options = ['NaCl', 'KCl']
    valid_pbc_options = ['cubic', 'box', 'truncated_octahedral']

    if data['protein_ff'] not in valid_protein_force_fields:
        print(f'  {bcolors.FAIL}Error:{bcolors.ENDC} protein force field {data["protein_ff"]} is not supported.\n'
                         f'    Valid options are one of {valid_protein_force_fields}')

    if len(data['non_standard_residues']) > 0:
        if data['ligand_ff'] not in valid_ligand_force_fields:
            print(f'  {bcolors.FAIL}Error:{bcolors.ENDC} ligand force field {data["ligand_ff"]} is not supported.\n'
                             f'    Valid options are one of {valid_ligand_force_fields}')

    if data['salt'] not in valid_ion_options:
        print(f'  {bcolors.FAIL}Error:{bcolors.ENDC} salt option "{data['salt']}" not supported.\n'
                         f'    Valid options are one of {valid_ion_options}')

    if data['pbc'] not in valid_pbc_options:
        print(f'  {bcolors.FAIL}Error: {bcolors.ENDC}pbc option "{data['pbc']}" not supported.\n'
                         f'    Valid options are one of {valid_pbc_options}')
        success = False
    return data, success

def validate_source(data):
    """
    Check source file is valid
    """
    success = True

    if not Path(data['source']).exists():
        print(f'  {bcolors.FAIL}Error:{bcolors.ENDC} cannot find file {data['source']}')
        success = False

    heavypdb, nonprotpdb, log = pdb4amber(data['source'])

    if not 'atom(s)\n\nNone' in log.read_text():
        print(f'  {bcolors.FAIL}Error:{bcolors.ENDC} missing protein atoms in source file - cannot continue.')
        print("  here is the log from pdb4amber:")
        print(log.read_text())
        success = False
    if 'gap of' in log.read_text():
        print(f'  {bcolors.FAIL}Error:{bcolors.ENDC} protein chain in source file has gaps - cannot continue.')
        print("  here is the log from pdb4amber:")
        print(log.read_text())
        success = False 

    tmp = mdt.load(nonprotpdb, standard_names=False)
    
    defined_resnames = list(data['non_standard_residues'].keys())
    used_resnames = set([r.name for r in tmp.topology.residues])
    for resname in defined_resnames:
        if resname not in used_resnames:
            print(f'  {bcolors.WARNING}Warning:{bcolors.ENDC} input yaml file defines non-standard residue {resname} but it is not present in the source file.')
            data['non_standard_residues'].pop(resname)
    
    for resname in used_resnames:
        if resname not in defined_resnames:
            print(f'  {bcolors.FAIL}Error:{bcolors.ENDC} source file contains residue {resname} not defined in input yaml file.')
            success = False
    return success


def adjust_protein_residue_names(data):
    """
    Adjust names of titratable and tautomeric residues
    """
    heavypdb, nonprotpdb, log = pdb4amber(data['source'])
    preppdb, log = pdb2pqr(heavypdb, data['target_pH'])
    finalpdb, _, log2 = pdb4amber(preppdb)

    t_orig = mdt.load(heavypdb, standard_names=False)
    t_final = mdt.load(finalpdb, standard_names=False)
    original_residue_names = [f"{r.name}{r.resSeq}{r.chain.chain_id}" for r in t_orig.topology.residues]
    final_residue_names = [f"{r.name}{r.resSeq}{r.chain.chain_id}" for r in t_final.topology.residues]
    first = True
    for o_name, f_name in zip(original_residue_names, final_residue_names):
        if o_name[:-1] != f_name[:-1]:
            if first:
                print('  Modified amino acid residue names:')
                first=False
            print(f"    {o_name} -> {f_name}")
    return finalpdb, nonprotpdb


def parameterize_nonstandard_residues(nonprotpdb, data):
    """
    Parameterize non-standard residues using the chosen force field
    """

    success = True
    lig_mol2files = []
    lig_frcmodfiles = []
    fh = FileHandler()
    t = mdt.load(nonprotpdb)
    

    # Set up logging to suppress RDKit warnings
    logger = logging.getLogger('rdkit')
    logger.handlers[0].setLevel(logging.ERROR)
    
    for ligand_resname in data['non_standard_residues']:
        mol2filepath = Path(f"{data['system_name']}/{ligand_resname}_{data['ligand_ff']}.mol2")
        if mol2filepath.exists():
            print(f"  {bcolors.WARNING}Warning:{bcolors.ENDC} mol2 file for {ligand_resname} already exists, skipping antechamber.")
            lig_mol2files.append(mol2filepath)
            continue
        ligand_smiles = data['non_standard_residues'][ligand_resname]
        template = Chem.MolFromSmiles(ligand_smiles)
        if template is None:
            print(f"  {bcolors.FAIL}Error:{bcolors.ENDC} RDKit could not interpret the SMILES provided for {ligand_resname}.")
            success = False
            return None, None, success
        sel = t.topology.select(f"resname {ligand_resname}")
        if len(sel) == 0:
            print(f"  {bcolors.FAIL}Error:{bcolors.ENDC} cannot find a residue with name {ligand_resname} in the structure.")
            success = False
            return None, None, success
        else:
            tlig = t.atom_slice(sel)
            sel = tlig.topology.select("resid 0")
            tlig = tlig.atom_slice(sel)
            ligf = fh.create(f'{ligand_resname}.pdb')
            ligf.write_text('')
            tlig.save(ligf)

            pose = Chem.MolFromPDBFile(str(ligf), sanitize=False, removeHs=False, proximityBonding=True)
            if pose is None:
                print(f"  {bcolors.FAIL}Error:{bcolors.ENDC} RDKit could not read the predicted pose for non_standard residue {ligand_resname}.")
                success = False
                return None, None, success
            pose.UpdatePropertyCache(strict=False)
            try:
                ligand_chemistry = AllChem.AssignBondOrdersFromTemplate(template, pose)
            except Exception as e:
                print(f"  {bcolors.FAIL}Error:{bcolors.ENDC} structure for {ligand_resname} in source does not match SMILES provided.")
                success = False
                return None, None, success
            ligand_with_h = Chem.AddHs(ligand_chemistry, addCoords=True)
            formal_charge = Chem.GetFormalCharge(ligand_with_h)
            molfile = fh.create('lig.mol')
            molfile.write_text(Chem.MolToMolBlock(ligand_with_h))
            mol2file, log = antechamber(molfile, ligand_resname, data['ligand_ff'], formal_charge)
            if 'Please check the total charge' in log.read_text():
                print(f"{bcolors.WARNING}log.read_text(){bcolors.ENDC}")

            ligand_atom_names = [a.name for a in tlig.topology.atoms]
            corrected_mol2 = ''
            fix = False
            n_fixed = 0
            for line in mol2file.read_text().split('\n'):
                if fix:
                    l = f"{line[:8]}{ligand_atom_names[n_fixed]:4s}{line[12:]}\n"
                    n_fixed += 1
                else:
                    l = line + '\n'
                corrected_mol2 += l
                if n_fixed >= len(ligand_atom_names):
                    fix = False
                if '@<TRIPOS>ATOM' in line:
                    fix = True
            mol2file.write_text(corrected_mol2)
            lig_mol2files.append(mol2file)

    for ligand_resname, mol2file in zip(data['non_standard_residues'], lig_mol2files):
        frcmodfile = Path(f"{data['system_name']}/{ligand_resname}_{data['ligand_ff']}.frcmod")
        if frcmodfile.exists():
            print(f"  {bcolors.WARNING}Warning:{bcolors.ENDC} frcmod file for {ligand_resname} already exists, skipping parmchk2.")
            lig_frcmodfiles.append(frcmodfile)
        else:
            frcmod = parmchk2(mol2file, data['ligand_ff'])
            lig_frcmodfiles.append(frcmod)
            txt = frcmod.read_text()
            offset = 0
            pen = 0.0
            while 'penalty score=' in txt[offset:]:
                offset += txt[offset:].index('penalty score=') + 14
                pen += float(txt[offset:offset+4])
            print(f"{bcolors.WARNING}  Note:{bcolors.ENDC} Parameterization penalty for {ligand_resname}: {pen}")
    return lig_mol2files, lig_frcmodfiles, success


def run_leap(finalpdb, data, lig_mol2files, lig_frcmodfiles, n_cations=0, n_anions=0):
    protein_ff = data['protein_ff']
    ligand_ff = data['ligand_ff']
    water_model = data['water_model']

    tleap_template = f"source leaprc.protein.{protein_ff}\n"
    tleap_template += f"source leaprc.{ligand_ff}\n"
    tleap_template += f"source leaprc.water.{water_model}\n"

    for ligand_resname in data['non_standard_residues']:
        tleap_template += f"loadAmberParams {ligand_resname}_{ligand_ff}.frcmod\n"
        tleap_template += f"{ligand_resname} = loadMol2 {ligand_resname}_{ligand_ff}.mol2\n"

    tleap_template += "system = loadpdb in.pdb\n"
    if data['pbc'] == 'truncated_octahedral':
        tleap_template += f"solvateOct system {water_model}BOX {data['padding']}\n"
    elif data['pbc'] == 'box':
        tleap_template += f"solvateBox system {water_model}BOX {data['padding']}\n"
    elif data['pbc'] == 'cubic':
        tleap_template += f"solvateBox system {water_model}BOX {data['padding']} iso \n"

    if data['salt'] == 'NaCl':
        cation = 'Na+'
        anion = 'Cl-'
    elif data['salt'] == 'KCl':
        cation = 'K+'
        anion = 'Cl-'
    if n_cations > 0 or n_anions > 0:
        tleap_template += f"addions system {cation} {n_cations}\naddions system {anion} {n_anions}\n"

    tleap_template += 'charge system\ncheck system\nsaveamberparm system out.prmtop out.rst7\nsavePdb system out.pdb\nquit'
    fh = FileHandler()
    tleapin = fh.create('tleap.in')
    tleapin.write_text(tleap_template)

    inputs = ['tleap.in', 'in.pdb']
    for ligand_resname in data['non_standard_residues']:
        inputs.append(f"{ligand_resname}_{ligand_ff}.mol2")
    for ligand_resname in data['non_standard_residues']:
        inputs.append(f"{ligand_resname}_{ligand_ff}.frcmod")
    tleap.set_inputs(inputs)
    
    rst7, prmtop, outpdb, leaplog = tleap(tleapin, finalpdb, *lig_mol2files, *lig_frcmodfiles)
    if 'Unit is OK' not in leaplog.read_text():
        print(f'  {bcolors.FAIL}Warning: Leap Failed!{bcolors.ENDC}')
    return rst7, prmtop, outpdb, leaplog, tleapin


def calculate_ion_parameters(outpdb, leaplog, data):
    """
    Calculate the numbers of ions that need to be added
    """
    tmp = mdt.load(outpdb)
    n_waters = len(tmp.topology.select('resname HOH')) // 3

    for line in leaplog.read_text().split('\n'):
        if 'Total unperturbed charge' in line:
            charge = np.rint(float(line.split()[3])).astype(int)

    ion_molarity = data['target_ion_molarity']
    n_ions = np.rint(n_waters * ion_molarity / 55.56).astype(int)

    if charge > 0:
        n_cations = n_ions - charge
        n_anions = n_ions
    else:
        n_cations = n_ions
        n_anions = n_ions + charge

    return n_cations, n_anions


def run_yaspt(yaml_file):
    """
    Run the YASPT pipeline
    """
    cmds = ['pdb4amber', 'pdb2pqr30', 'antechamber', 'parmchk2', 'tleap']
    for cmd in cmds:
        if not installed(cmd):
            print(f"  {bcolors.FAIL}Error:{bcolors.ENDC} Required command {cmd} is not installed or not in PATH.")
            sys.exit(1)

    print(f"Checking input yaml file {yaml_file}:")
    validated_data, success = validate_yaml_input(yaml_file)
    if not success:
        sys.exit(1)

    system_name = validated_data['system_name']
    output_dir = Path(system_name)
    if not output_dir.exists():
        output_dir.mkdir()

    validated_yaml_file = output_dir / Path(yaml_file).name
    validated_yaml_file.write_text(yaml.dump(validated_data, sort_keys=False))
    print(f"  {bcolors.OKGREEN}Validated YAML file written to {validated_yaml_file}{bcolors.ENDC}\n")

    print(f"Checking source file {validated_data['source']} using {bcolors.BOLD}pdb4amber{bcolors.ENDC}:")
    success = validate_source(validated_data)
    if not success:
        sys.exit(1)
    else:
        print(f"{bcolors.OKGREEN}  No issues\n{bcolors.ENDC}")

    print(f"Preparing protein structure using {bcolors.BOLD}pdb2pqr{bcolors.ENDC} and {bcolors.BOLD}propka{bcolors.ENDC}:")
    final_pdb, nonprot_pdb = adjust_protein_residue_names(validated_data)
    final_pdb_file = output_dir / Path(f"{system_name}.pdb")
    final_pdb.save(final_pdb_file)
    print(f"{bcolors.OKGREEN}  Leap-ready PDB file written to {final_pdb_file}{bcolors.ENDC}\n")
    print(f"Preparing non-standard residues using {bcolors.BOLD}antechamber{bcolors.ENDC} and {bcolors.BOLD}parmchk2{bcolors.ENDC}:")
    lig_mol2files, lig_frcmodfiles, success = parameterize_nonstandard_residues(nonprot_pdb, validated_data)
    if not success:
        sys.exit(1)

    lig_names = [f"{lig}_{validated_data['ligand_ff']}" for lig in validated_data['non_standard_residues']]
    for lig_mol2file, lig_name in zip(lig_mol2files, lig_names):
        if not isinstance(lig_mol2file, Path):
            lig_mol2file.save(output_dir / f"{lig_name}.mol2")
            print(f"{bcolors.OKGREEN}  Created mol2 file {output_dir / f'{lig_name}.mol2'}{bcolors.ENDC}")
    for lig_frcmodfile, lig_name in zip(lig_frcmodfiles, lig_names):
        if not isinstance(lig_frcmodfile, Path):
            lig_frcmodfile.save(output_dir / f"{lig_name}.frcmod")
            print(f"{bcolors.OKGREEN}  Created frcmod file {output_dir / f'{lig_name}.frcmod'}{bcolors.ENDC}")

    print(f"\nGenerating coordinate and topology files using {bcolors.BOLD}tleap{bcolors.ENDC}:")
    rst7, prmtop, ambpdb, leaplog, _ = run_leap(final_pdb, validated_data, lig_mol2files, lig_frcmodfiles)

    n_cations, n_anions = calculate_ion_parameters(ambpdb, leaplog, validated_data)
    if min(n_cations, n_anions) < 0:
        print(f'  {bcolors.FAIL}Error:{bcolors.ENDC} the target ion concentration cannot be achieved'
            f'- make it lower or increase the size of the box.')
        sys.exit(1)

    rst7, prmtop, ambpdb, log, tleapin = run_leap(final_pdb, validated_data, lig_mol2files,
                                        lig_frcmodfiles, n_cations=n_cations, n_anions=n_anions)

    rst7_file = output_dir / f"{system_name}.rst7"
    print(f"{bcolors.OKGREEN}  Created restart file {rst7_file}")
    prmtop_file = output_dir / f"{system_name}.prmtop"
    print(f"  Created topology file {prmtop_file}")
    tleap_in_file = output_dir / f"{system_name}_tleap.in"
    tleap_text = tleapin.read_text()
    tleap_text = tleap_text.replace('in.pdb', f"{system_name}.pdb")
    tleap_text = tleap_text.replace('out.prmtop', f"{system_name}.prmtop")
    tleap_text = tleap_text.replace('out.rst7', f"{system_name}.rst7")
    tleap_text = tleap_text.replace('out.pdb', f"{system_name}_amber.pdb")
    tleapin.write_text(tleap_text)
    tleapin.save(tleap_in_file)
    print(f"  Saved leap input file as {tleap_in_file}")
    tleap_log_file = output_dir / f"{system_name}_tleap.log"
    print(f"  Saved leap log file as {tleap_log_file}{bcolors.ENDC}\n")
    rst7.save(rst7_file)
    prmtop.save(prmtop_file)
    log.save(tleap_log_file)

         
