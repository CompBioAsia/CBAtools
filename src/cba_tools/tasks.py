from crossflow.tasks import SubprocessTask

which = SubprocessTask('which {command}')
which.set_inputs(['command'])
which.set_outputs(['DEBUGINFO'])


pdb4amber = SubprocessTask(
    'pdb4amber'
    ' -i in.pdb'
    ' -o out.pdb'
    ' -y'
    ' -l out.log'
)
pdb4amber.set_inputs(['in.pdb'])
pdb4amber.set_outputs(['out.pdb', 'out_nonprot.pdb', 'out.log'])


pdb2pqr = SubprocessTask(
    'pdb2pqr30'
    ' --ff AMBER'
    ' --keep-chain'
    ' --ffout AMBER'
    ' --pdb-output out.pdb'
    ' --titration-state-method propka'
    ' --with-ph {pH}'
    ' -q'
    ' in.pdb'
    ' out.pqr'
    ' > out.log 2>&1'
)
pdb2pqr.set_inputs(['in.pdb', 'pH'])
pdb2pqr.set_outputs(['out.pdb', 'out.log'])

ek = "-ek 'qm_theory=\"AM1\", grms_tol=0.0005,"
ek += " scfconv=1.d-10, maxcyc=0'"

antechamber = SubprocessTask('antechamber'
                             ' -i lig.mol'
                             ' -fi mdl'
                             ' -o lig.mol2'
                             ' -fo mol2'
                             ' -at {gafftype}'
                             ' -c bcc'
                             ' -nc {charge}'
                             ' -rn {ligname}'
                             ' -pf y {ek}'
                             ' > logfile &2>1')
antechamber.set_inputs(['lig.mol', 'ligname', 'gafftype', 'charge', 'ek'])
antechamber.set_outputs(['lig.mol2', 'logfile'])
antechamber.set_constant('ek', ek)

parmchk2 = SubprocessTask('parmchk2'
                          ' -i lig.mol2'
                          ' -f mol2'
                          ' -o lig.frcmod'
                          ' -s {gafftype}')
parmchk2.set_inputs(['lig.mol2', 'gafftype'])
parmchk2.set_outputs(['lig.frcmod'])

tleap = SubprocessTask('tleap -f tleap.in')
tleap.set_outputs(['out.rst7', 'out.prmtop', 'out.pdb', 'leap.log'])