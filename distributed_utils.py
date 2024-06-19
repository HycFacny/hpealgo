from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
import multiprocessing as mp
from contextlib import contextmanager
import numpy as np

import torch
from torch.utils.data.sampler import Sampler


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indices sequentially, making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training), which means that the model params won't
    have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add
    extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather`
    or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert (
            len(indices) == self.total_size
        ), f"Indices length {len(indices)} and total size {self.total_size} mismatched"

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert (
            len(indices) == self.num_samples
        ), f"Indices length {len(indices)} and sample number {self.num_samples} mismatched"

        return iter(indices)

    def __len__(self):
        return self.num_samples


def distributed_concat(tensor, num_samples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_samples]


def distributed_concat_non_tensor(x, num_samples, queue):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    queue.put(x)
    torch.distributed.barrier()
    
    if rank != 0:
        return None
    
    output = []
    for _ in range(world_size):
        output.append(queue.get(True))
    output = np.concatenate(output, axis=0)
    return output[:num_samples]


def starter(fn, cfg, args, barrier):
    queue = mp.Queue()
    world_size = len(cfg.GPUS)
    
    processes = []
    for i in range(world_size):
        process = mp.Process(
            target=fn,
            args=(i, cfg, args, world_size, queue, barrier)
        )
        process.start()
        processes.append(process)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    if local_rank != 0:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def MainProcess(rank, fn, *args):
    x = None
    if rank != 0:
        print(torch.distributed.get_rank())
        torch.distributed.barrier()
    else:
        x = fn(*args)
        print(torch.distributed.get_rank())
        torch.distributed.barrier()
    return x
