import torch
import torch.distributed.rpc as rpc

if __name__ == '__main__':
    rpc.init_rpc("worker0", rank=0, world_size=2)
    ret = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(2),3))
    rpc.shutdown()
