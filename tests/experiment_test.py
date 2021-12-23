# -*- coding: utf-8 -*-


from .context import recoprot


def test_configurations():

    data = {
        "dtype": "atoms",
        "bert": False,
        "database": "/tmp",
        "n_epochs": [10, 25, 50],
        "learning_rate": [0.001, 0.005],
        "conv_filters": [[128], [128, 256, 512]],
        "dense_filters": [[256]],
        "alpha": False
    }
    confs = recoprot.Configurations(data)
    ref = [
        recoprot.Configuration(False, 10, 0.001, [128], [256], False),
        recoprot.Configuration(False, 10, 0.001, [128, 256, 512], [256], False),
        recoprot.Configuration(False, 10, 0.005, [128], [256], False),
        recoprot.Configuration(False, 10, 0.005, [128, 256, 512], [256], False),
        recoprot.Configuration(False, 25, 0.001, [128], [256], False),
        recoprot.Configuration(False, 25, 0.001, [128, 256, 512], [256], False),
        recoprot.Configuration(False, 25, 0.005, [128], [256], False),
        recoprot.Configuration(False, 25, 0.005, [128, 256, 512], [256], False),
        recoprot.Configuration(False, 50, 0.001, [128], [256], False),
        recoprot.Configuration(False, 50, 0.001, [128, 256, 512], [256], False),
        recoprot.Configuration(False, 50, 0.005, [128], [256], False),
        recoprot.Configuration(False, 50, 0.005, [128, 256, 512], [256], False),
    ]
    calc = [i for i in confs]
    assert ref == calc
    return
