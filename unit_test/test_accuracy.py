from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pathlib import Path
import numpy as np

import torch
from torch.tensor import Tensor

import _init_paths
from core.evaluate import get_accuracy


def test_accuracy():
    hmp_pred = torch.arange(0, 1 * 17 * 64 * 48).view(1, 17, 64, 48).numpy()
    hmp_gt = torch.arange(1, 1 * 17 * 64 * 48 + 1).view(1, 17, 64, 48).numpy()
    print(hmp_pred)
    print(hmp_gt)
    
    _, avg_acc, cnt, b = get_accuracy(hmp_pred, hmp_gt)
    print(avg_acc)


if __name__ == '__main__':
    test_accuracy()
