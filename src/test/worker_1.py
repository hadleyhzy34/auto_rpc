import torch
import torch.distributed.rpc as rpc

rpc.init_rpc("worker2",rank=1,world_size=2)
rpc.shutdown()
