"""run.py:"""
#!/usr/bin/env python
import os
import time
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run1(rank, size):
    """ Distributed function to be implemented later. """
    while True:
        time.sleep(2)
        print(f'process {rank} is running')

def run2(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

def run3(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run3))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
