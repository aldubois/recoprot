# -*- coding: utf-8 -*-

"""
Experiment runner.
"""

# Standard Library
import os
import logging
import argparse
from collections import namedtuple
from itertools import product

# External dependencies
import yaml
import torch
import pandas as pds

# Internals
from .symbols import DEVICE
from .data import (
    AtomsTrainingDataset,
    AtomsValidationDataset,
    ResiduesTrainingDataset,
    ResiduesValidationDataset
)
from .nn import AtomsNetwork, ResiduesNetwork
from .train import train, evaluate

DTYPE = "dtype"
BERT = "bert"
DB = "database"
N_EPOCHS = "n_epochs"
LR = "learning_rate"
CONV = "conv_filters"
DENSE = "dense_filters"

DF_N_EPOCHS = "N Epochs"
DF_LR = "Learning rate"
DF_CONV = "Conv filters"
DF_DENSE = "Dense filters"
DF_AUC_TRAINING = "AUC Training"
DF_AUC_VALIDATION = "AUC Validation"
DISTANCE = "distance"

DF_COLUMNS = [DF_N_EPOCHS, DF_LR, DF_CONV, DF_DENSE, DF_AUC_TRAINING,
              DF_AUC_VALIDATION, DISTANCE]


Configuration = namedtuple(
    "Configuration",
    ["bert", "n_epochs", "lr",
     "convs", "dense", "alpha"]
)


class Configurations:

    """
    Experiences configurations for each hyperparameters.
    """
    def __init__(self, data):
        self.dtype = data[DTYPE]
        self.bert = data[BERT]
        self.database = data[DB]
        self.n_epochs = data[N_EPOCHS]
        self.learning_rates = data[LR]
        self.conv_filters = data[CONV]
        self.dense_filters = data[DENSE]
        self.alpha = data[DISTANCE]

    def __repr__(self):
        return (f"Configurations(dtype={self.dtype},"
                f" bert={self.bert}"
                f" database={self.database}"
                f" n_epochs={self.n_epochs},"
                f" learning_rates={self.learning_rates},"
                f" conv_filters={self.conv_filters},"
                f" dense_filters={self.dense_filters},"
                f" alpha={self.alpha})")
        
    def __iter__(self):
        return (
            Configuration(*data)
            for data in product([self.bert],
                                self.n_epochs,
                                self.learning_rates,
                                self.conv_filters,
                                self.dense_filters)
        )


def experiment_main():
    """
    Main function for steuptools entrypoint.
    """
    configurations, output_file = parse_experiment_args()
    logging.info(f"{configurations}")

    df = pds.DataFrame(columns=DF_COLUMNS)

    experiment = atoms_experiment if configurations.dtype == "atoms" else residues_experiment
    
    for config in configurations:
        res = experiment(configurations.database, config)
        df = df.append(res, ignore_index=True)

    df.to_csv(output_file)
    return
        

def atoms_experiment(database, config):
    # Training
    training_set = AtomsTrainingDataset(database)
    gnn = AtomsNetwork(config.convs, config.dense, config.bert)
    model = gnn.to(DEVICE)
    losses = train(
        model,
        training_set,
        config.n_epochs,
        config.lr,
    )

    res = {
        DF_N_EPOCHS: config.n_epochs,
        DF_LR: config.lr,
        DF_CONV: config.convs,
        DF_DENSE: config.dense
    }
    logging.info("Training set: ")
    res[DF_AUC_TRAINING] = evaluate(model, training_set)
    logging.info("    AUC: %.4f" % (res[DF_AUC_TRAINING]))

    
    # Validation
    validation_set = AtomsValidationDataset(database)
    logging.info("Validation set: ")
    res[DF_AUC_VALIDATION] = evaluate(model, validation_set)
    logging.info("    AUC: %.4f %%" % (res[DF_AUC_VALIDATION]))
    return res


def residues_experiment(database, config):
    # Training
    training_set = ResiduesTrainingDataset(database)
    gnn = ResiduesNetwork(config.convs, config.dense, config.bert)
    model = gnn.to(DEVICE)
    losses = train(
        model,
        training_set,
        config.n_epochs,
        config.lr,
    )

    res = {
        DF_N_EPOCHS: config.n_epochs,
        DF_LR: config.lr,
        DF_CONV: config.convs,
        DF_DENSE: config.dense
    }
    logging.info("Training set: ")
    res[DF_AUC_TRAINING] = evaluate(model, training_set)
    logging.info("    AUC: %.4f" % (res[DF_AUC_TRAINING]))

    
    # Validation
    validation_set = ResiduesValidationDataset(database)
    logging.info("Validation set: ")
    res[DF_AUC_VALIDATION] = evaluate(model, validation_set)
    logging.info("    AUC: %.4f %%" % (res[DF_AUC_VALIDATION]))
    return res


def parse_experiment_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser(description='Launch experiment.')

    parser.add_argument(
        "-c", "--conf", dest="conf",
        required=True,
        type=_conf,
        help="Configuration file of the experiment"
    )

    parser.add_argument(
        "-o", "--output-file", dest="output_file",
        required=True,
        type=_output_file,
        help="CSV output file giving the AUC for each configuration."
    )
    
    # Optional arguments
    parser.add_argument("--info", dest="log",
                        action="store_true",
                        default=False,
                        help="Display information messages")

    args = parser.parse_args()

    log_fmt = '%(levelname)7s: %(message)s'
    if args.log:
        logging.basicConfig(format=log_fmt, level=logging.INFO)
    else:
        logging.basicConfig(format=log_fmt)
    
    return _verify_conf(args.conf), args.output_file


def _conf(x):

    if not os.path.exists(x):
        raise argparse.ArgumentTypeError(f"{x} is not a valid path")

    return x


def _verify_conf(x):
    
    with open(x) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError(f"{x} should contain pair of key/values!\n{data}")

    for key in [DTYPE, DB, N_EPOCHS, LR, CONV, DENSE]:
        if key not in data:
            raise argparse.ArgumentTypeError(f"{x} should contain the key {key}!\n{data}")

    if data[DTYPE] not in ("atoms", "residues"):
        raise argparse.ArgumentTypeError("The data type needs to be 'atoms' or 'residues'")
        
    _is_dir(data[DB])
        
    for key in [N_EPOCHS, LR, CONV, DENSE]:
        if not isinstance(data[key], list):
            raise argparse.ArgumentTypeError(f"The data {key} in the configuration file"
                                             f" should be a list and is a  {type(data[key])}")

    for n in data[N_EPOCHS]:
        _is_positive_int(n)

    for lr in data[LR]:
        _learning_rate(lr)

    for conv in data[CONV]:
        if not isinstance(conv, list):
            raise argparse.ArgumentTypeError(f"The data {CONV} in the configuration file"
                                             f" should be a list of list but"
                                             f" contains a {type(conv)}")
        for filt in conv:
            _is_positive_int(filt)

    for dense in data[DENSE]:
        if not isinstance(dense, list):
            raise argparse.ArgumentTypeError(f"The data {DENSE} in the configuration file"
                                             f" should be a list of list but"
                                             f" contains a {type(dense)}")
        for filt in dense:
            _is_positive_int(filt)

    return Configurations(data)
            

def _learning_rate(x):
    """
    Verification of the learning rate argument.
    """
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))


def _is_dir(arg):
    """
    Verify that the argument is an existing directory.
    """
    if not os.path.isdir(arg):
        raise argparse.ArgumentTypeError(f"The input directory %s does not exist! {arg}")


def _is_positive_int(x):
    """
    Verify that the argument is a positive integer.
    """
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} is not an integer literal")

    if x <= 0.0:
        raise argparse.ArgumentTypeError(f"{x} isn't a positive number")

def _output_file(x):

    if os.path.exists(x):
        raise argparse.ArgumentTypeError(f"The output file {x} already exist!")

    if not os.path.isdir(os.path.dirname(os.path.abspath(x))):
        raise argparse.ArgumentTypeError(f"The directory of the output file {x} doesn't exist!")
    return x
