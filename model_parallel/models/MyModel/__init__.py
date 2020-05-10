# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage2_1 import Stage2_1
from .stage3 import Stage3
from .mymodel import MyModel

def arch():
    return "mumodel"

def model(criterion):
    return [
        (Stage0(), ["input"], ["out0", "out1"]),
        (Stage1(), ["out0", "out1"], ["out3", "out2"]),
        (Stage2(), ["out3", "out2"], ["out4", "out5"]),
        (Stage2_1(), ["out4", "out5"], ["out7", "out8"]),
        (Stage3(), ["out7", "out8"], ["out6"]),
        (criterion, ["out6"], ["loss"])
    ]

def full_model():
    return MyModel()
