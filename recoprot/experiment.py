# -*- coding: utf-8 -*-

"""
Experiment runner.
"""

# Standard Library
import os
import logging
import argparse

# External dependencies
import torch

# Internals
from .symbols import DEVICE
from .data import TrainingDataset, ValidationDataset
from .nn import CompleteNetwork
from .train import train, evaluate


class ExperimentOptions:

    def __init__(self, database, learning_rate, n_epochs):
        self.database = database
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        
    def __repr__(self):
        return (f"ExperimentOptions(database={self.database}"
                f" learning_rate={self.learning_rate},"
                f" n_epochs={self.n_epochs})")
        

def experiment_main():
    """
    Main function for steuptools entrypoint.
    """
    options = parse_experiment_args()
    logging.info(f"{options}")

    # Training
    training_set = TrainingDataset(options.database)
    gnn = CompleteNetwork([128, 256])
    model = gnn.to(DEVICE)
    losses = train(
        model,
        training_set,
        options.n_epochs,
        options.learning_rate,
    )

    # Validation
    validation_set = ValidationDataset(options.database)
    logging.info("Validation set: ")
    auc = evaluate(model, validation_set)
    logging.info("    AUC: %.4f %%" % (auc))



def parse_experiment_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser(description='Launch experiment.')

    parser.add_argument(
        "--lr", default=0.001,
        dest="learning_rate",
        type=_learning_rate,
        help='Learning rate'
    )
    parser.add_argument(
        "-d", "--database", dest='database',
        required=True, metavar='DIR',
        type=_is_dir,
        help='Data output directory'
    )
    parser.add_argument(
        "-n", "--n_epochs", default=10,
        dest="n_epochs",
        type=_n_epochs,
        help="Number of epochs for the training phase"
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
    
    return ExperimentOptions(args.database, args.learning_rate, args.n_epochs)


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
    return x


def _is_dir(arg):
    """
    Verify that the argument is an existing directory.
    """
    if not os.path.isdir(arg):
        raise argparse.ArgumentTypeError(f"The input directory %s does not exist! {arg}")
    return arg


def _n_epochs(x):
    """
    Verify that the argument is a positive integer.
    """
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%d not an integer literal" % (x,))

    if x < 0.0:
        raise argparse.ArgumentTypeError("%d isn't a positive number" % (x,))
    return x
