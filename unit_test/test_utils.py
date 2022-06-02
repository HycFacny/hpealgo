from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import _init_paths
_init_paths.add_lib_paths()

print(sys.path)
import .utils.utils


def test_util_package():
    print(1)

if __name__ == '__main__':
    test_util_package()