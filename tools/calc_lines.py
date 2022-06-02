from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pathlib import Path

import _init_paths


def get_projroot():
    root = Path.cwd()
    while str(root).split('/')[-1] != 'hpealgo':
        root = root.parent
    return str(root)


if __name__ == '__main__':
    root = Path(get_projroot())
    cnt = 0
    
    all_files = root.rglob('*.py')

    for item in all_files:
        with open(str(item), 'r') as f:
            lines = f.readlines()
            cnt += len(lines)
    
    print(cnt)