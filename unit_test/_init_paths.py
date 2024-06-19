from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from pathlib import Path


def fold_valid(path):
    name = str(path).split('/')[-1]
    # print(name)
    if '.' in name or '__' in name:
        return False
    useless = ['data', 'output', 'coco', 'mpii', 'results',
        'experiments', 'log', 'Makefile']
    if name in useless: return False
    return True

def get_folds_recursive(root):
    lists = list(root.rglob('*'))
    folds = [ str(item) for item in lists if fold_valid(item) ]
    return folds

def add_path(path):
    if not path in sys.path: sys.path.insert(0, path)

def add_lib_paths():
    project_root = Path.cwd()
    while str(project_root).split('/')[-1] != 'hpealgo':
        project_root = project_root.parent
    # print(project_root)

    lib_root = project_root / 'lib'
    # lib_fold = get_folds_recursive(lib_root)

    tools_root = project_root / 'tools'
    # tools_fold = get_folds_recursive(tools_root)

    unittest_root = project_root / 'unit_test'
    # unittest_fold = get_folds_recursive(unittest_root)

    paths = [str(lib_root), str(tools_root), str(unittest_root)]
    # paths = []
    # paths.extend(lib_fold)
    # paths.extend(tools_fold)
    # paths.extend(unittest_fold)

    for path in paths: add_path(path)
    # print(sys.path)

add_lib_paths()
