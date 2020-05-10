import torch
from torch import distributed as dist
import os
import numpy as np
import datetime, time
os.environ["CUDA_VISIBLE_DEVICE"] = '0'

master_addr = '172.19.0.16'
master_port = 10000
world_size = 2
rank = 1
backend = 'nccl'
torch.cuda.set_device('cuda:0')
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = str(master_port)
os.environ['WORLD_SIZE'] = str(world_size)
os.environ['RANK'] = str(rank)
dist.init_process_group(backend, init_method='tcp://172.19.0.16:10000', timeout=datetime.timedelta(0,10),rank=rank,world_size=world_size)
#dist.init_process_group(backend, rank=rank, world_size=world_size)
print("Finished initializing process group; backend: %s, rank: %d, "
	"world_size: %d" % (backend, rank, world_size))

group0 = dist.new_group(ranks=[1], timeout=datetime.timedelta(0, 180), backend='nccl')
group = dist.new_group(ranks=[0, 1], timeout=datetime.timedelta(0, 180), backend='nccl')

a = torch.from_numpy(np.random.rand(3, 3)).cuda()
print("Start broadcasting: ", time.time())
dist.broadcast(tensor=a, src=1, group=group)
#dist.recv(a)
print("Finish broadcasting: ", time.time())
print(a)

dist.broadcast(tensor=a, src=0, group=group)
print(a)