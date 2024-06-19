from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
sys.path.append(str(Path.cwd().parent))

import tools._init_paths


def get_project_root():
    root = Path.cwd()
    while (str(root).split('/')[-1] != 'hpealgo'):
        root = root.parent
    return str(root)

def get_dataset_root(dataset):
    all_dataset_root = Path('/home/huangyuchao/projects/datasets')
    if dataset == 'coco': dataset += '2017'
    dataset_root = all_dataset_root / dataset
    return str(dataset_root)


if __name__ == '__main__':
    print(get_project_root())
    print(get_dataset_root('coco'))