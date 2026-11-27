# CBA-Tools
A package providing useful tools for Amber-focussed MD simulation setup.

Updated for CompBioAsia 2027.

Installing the package makes available a couple of command-line utilities:

#### yaspt

Provides a workflow that calls programs from **Ambertools** to complete all steps required to parameterize a biomolecular system, starting from a PDB format file of the solute.
```
usage: yaspt [-h] [-y YAML_FILE] [-v] [-t]

YASPT - Yet Another Simulation Preparation Tool

options:
  -h, --help            show this help message and exit
  -y, --yaml YAML_FILE  Input YAML file with system preparation parameters
  -v, --version         show program's version number and exit
  -t, --template        Output a template YAML file

```

#### mesmy

Generates the script and input files to run a multi-step MD equilibration workflow

```
usage: mesmy [-h] -i INPCRD -p PRMTOP [--version]

Create a script for a multi-step Amber MD relaxation/equilibration workflow.

options:
  -h, --help           show this help message and exit
  -i, --inpcrd INPCRD  Input Amber CRD file.
  -p, --prmtop PRMTOP  Input Amber PRMTOP file.
  --version            show program's version number and exit

```


## Authors

* Charlie Laughton charles.laughton@nottingham.ac.uk
* Bindu Kolli hima-bindu.kolli@stfc.ac.uk
