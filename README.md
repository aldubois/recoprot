# recoprot


## Installation procedure


### Clone the repository

```bash
git clone https://github.com/aldubois/recoprot.git
```


### Setup and enter your virtual environment

```bash
VIRTUALENV=/path/to/your/future/virtual/env
mkdir ${VIRTUALENV}
python3 -m venv ${VIRTUALENV}
source ${VIRTUALENV}/bin/activate
```

### Install the prerequisites

```bash
cd recoprot
make init
```


### Launch test base

```bash
make test
```


### Proceed with installation

```bash
python3 setup.py install
```


## Using the package

### Download the Docking Benchmark v5.

The data used to run the experiments is the [Docking Benchmark v5](https://zlab.umassmed.edu/benchmark/benchmark5.5.tgz).
Once the archived downloaded and uncompressed, the relative directory of interest is *benchmark5.5/structures*.



### Preprocess the data

To preprocess the the data, call:

```bash
export DB_SIZE=1000000000
preprocess -i /path/to/dbd/pdbfiles -o /path/for/output/lmdb/database -n ${DB_SIZE}
```

With *DB_SIZE* the memory size of the LMDB database that will be created.


### Train, validate and test models

TODO.



## Theoretical information

The goal of the models are to predict, for a ligand/receptor couple of
protein, which residues interact with each other.


### Data labelling

Two residues interact with each other if, on their bound structure,
the two residues are at a distance less than 6 Angstrom. This
information, for each couple residues of each couple of proteins in
the database, correspond to the target of our model.


### Model input features

For each proteins, for both the ligand and the receptor, the following
features are extracted:
- Atom encoding: each atoms are encoded with the following categories: [1, C, CA, CB, CG, CH2, N, NH2, O1, O2, OG,OH, SE]
- Residue encoding: each atoms corresponding residue is encoded with the following categories: [1, ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL]
- Atom neighbors encoding: information on the 10 closest neighbors of each atoms in the same residue and from different residues.


### Solving bound/unbound structure discrepancy

The list of residues in the bound and unbound structure of a protein is not necessarily the same. It means that we can get a number of samples in the labels different from the number of pairs of residues in the unbound structre.

To solve this problem, we run an alignment algorithm (here Needleman Wunsch) to extract the biggest common residues list between the bound and the unbound structure. The processing is then used only on this list of residues instead of on the whole protein chain.