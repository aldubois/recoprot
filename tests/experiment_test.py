# -*- coding: utf-8 -*-


from .context import recoprot


def test_configurations():

    data = {
        "database": "/tmp",
        "n_epochs": [10, 25, 50],
        "learning_rate": [0.001, 0.005],
        "conv_filters": [[128], [128, 256, 512]],
        "dense_filters": [[256]]
    }
    confs = recoprot.Configurations(data)
    ref = [
        recoprot.Configuration(10, 0.001, [128], [256]),
        recoprot.Configuration(10, 0.001, [128, 256, 512], [256]),
        recoprot.Configuration(10, 0.005, [128], [256]),
        recoprot.Configuration(10, 0.005, [128, 256, 512], [256]),
        recoprot.Configuration(25, 0.001, [128], [256]),
        recoprot.Configuration(25, 0.001, [128, 256, 512], [256]),
        recoprot.Configuration(25, 0.005, [128], [256]),
        recoprot.Configuration(25, 0.005, [128, 256, 512], [256]),
        recoprot.Configuration(50, 0.001, [128], [256]),
        recoprot.Configuration(50, 0.001, [128, 256, 512], [256]),
        recoprot.Configuration(50, 0.005, [128], [256]),
        recoprot.Configuration(50, 0.005, [128, 256, 512], [256]),
    ]
    calc = [i for i in confs]
    assert ref == calc
    return
