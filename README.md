# recoprot


## Installation procedure


### Clone the repository

git clone https://github.com/aldubois/recoprot.git


### Setup and enter your virtual environment

VIRTUALENV=/path/to/your/future/virtual/env
mkdir ${VIRTUALENV}
python3 -m venv ${VIRTUALENV}
source ${VIRTUALENV}/bin/activate


### Install the prerequisites

cd recoprot
make init


### Launch test base

make test


### Proceed with installation

python3 setup.py install