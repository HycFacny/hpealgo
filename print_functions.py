from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from pathlib import Path


debug_flag = {}
logger = logging.getLogger(__name__)


def print_inter_debug_info(item_name, item, testpart):
    """ print debug infomation to given file ' output_dir/unit_test/test_{}/test_{}_temporal_output.txt ' """
    global debug_flag

    if not testpart in debug_flag.keys():
        debug_flag[testpart] = False

    output_dir = Path.cwd()
    while str(output_dir).split('/')[-1] != 'hpealgo':
        output_dir = output_dir.parent
    output_dir = output_dir / 'unit_test' / 'test_{}'.format(testpart)
    if not output_dir.exists(): output_dir.mkdir()
    output_file = output_dir / 'test_{}_temporal_output.txt'.format(testpart)
    
    if not debug_flag[testpart]:
        debug_flag[testpart] = True
        with open(str(output_file), "w") as f:
            f.writelines('........................')

    with open(str(output_file), 'a') as f:
        f.write(item_name + ': >>>>\n'+ str(item) + '\n')


def print_inter_name_value(name_value, full_arch_name):
    """ print training intermediate results in markdown format """
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info('| Arch ' + ' '.join(['| {}'.format(name) for name in names]) + ' |')
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


def print_value_and_type(x, if_print=True):
    if if_print:
        print(x, type(x))
    return str(x) + '\n' + str(type(x))
