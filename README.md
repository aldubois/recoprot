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

### Preprocess the data

To preprocess the the data, call:

```bash
preprocess -i /path/to/dbd/pdbfiles -o /path/for/output/lmdb/database
```
