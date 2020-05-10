# coding=utf-8
from torchvision.models.resnet import ResNet, Bottleneck
import torchvision.transforms as transforms
import torchvision
import torch
from torch import distributed as dist
from torch import nn, optim
from torch.autograd import Variable
import time, os, datetime

from models import ResNet50
from runtime import runtime

import argparse
from socket import *
from time import ctime
import threading
import inspect
import ctypes


parser = argparse.ArgumentParser(description='PyTorch Model Parallel')
parser.add_argument('--rank', type=str, help='rank id')
parser.add_argument('--dist_addr', type=str, help='address of rank 0')
config_list = [None, './models/ResNet50/mp_conf_1gpu.json', './models/ResNet50/mp_conf_2gpu.json']
config = './models/ResNet50/mp_conf_1gpu.json'
next_epoch = True

def receive_message():
    global config_list
    global config
    global next_epoch
    while True:
        data = tcpCliSock.recv(1024);
        if not data:
            continue;
        data = data.decode('utf-8')
        print("\n" + str(time.time()))
        print("******data received: ", data)
        if data.split(":")[0] == "config":
            print("config for next epoch: ", data);
            config = config_list[int(data.split(":")[1])]
        else:
            next_epoch = True
            print("next_epoch: ", next_epoch)

def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


args = parser.parse_args()
#torch.cuda.set_device('cuda:' + args.rank)
master_addr = '172.19.0.5'
#dist_addr = '172.19.0.16'
dist_addr = args.dist_addr
print(args.dist_addr)
master_port = 10000
world_size = 2
backend = 'nccl'
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = str(master_port)
os.environ['WORLD_SIZE'] = str(world_size)
os.environ['RANK'] = str(args.rank)


# connect to master
Addr = (master_addr, 8080)
tcpCliSock = socket(AF_INET, SOCK_STREAM);
tcpCliSock.connect(Addr);
# thread to receive new config information
t = threading.Thread(target=receive_message)
t.start()



dist.init_process_group('nccl', init_method='tcp://'+dist_addr+':'+str(master_port),
    rank=int(args.rank), world_size=world_size)
print("initialized process group!")


transform = transforms.Compose([ transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

num_classes = 10
num_epoches = 7
num_batches = 3
batch_size = 100
image_w = 128
image_h = 128

train_data = torchvision.datasets.CIFAR10(root='/gpfs/share/home/1600012892/pytorch/model_parallel/CIFAR10data',
    train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
 shuffle=True)
test_data = torchvision.datasets.CIFAR10(root='/gpfs/share/home/1600012892/pytorch/model_parallel/CIFAR10data',
    train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
 shuffle=False)

loss_fn = nn.MSELoss
model = ResNet50.model(loss_fn)


def train(train_loader, r, epoch):

    epoch_start_time = time.time()
    print("Epoch %d start time: %.3f" % (epoch, epoch_start_time))

    for i, data in enumerate(train_loader, 0): 
        inputs, labels = data

        # perform forward pass
        r.run_forward(inputs, labels)

        # perform backward pass
        r.run_backward()

    #if args.rank == '0':
    #    #print("Epoch %d: %.3f loss: %.3f" % (epoch, losses.avg))
    #    print("Epoch start time: %.3f" % (epoch_start_time))
    print("Epoch %d: loss: %.3f" % (epoch, r.loss()))


# 初始化runtime
r = runtime.ModelParallelRuntime(model=model, rank=args.rank, world_size=2,
    config='./models/ResNet50/mp_conf_1gpu.json')

#optimizer = torch.optim.SGD(r.parameters(),lr=0.001)
#num_epoches

for epoch in range(7):
    while True:
        if next_epoch == True:
            next_epoch = False
            break

    r.reset_loss()
    # wait for signal from master
    if r.config != config:
        print("\n" + str(time.time()) + ": start scaling")
        r.scale(rank=args.rank, config=config)
        print("\n" + str(time.time()))
        print("finished scaling")

    train(train_loader, r, epoch)
    #print(r.loss())
    print("\n" + str(time.time()) + ": send signal to master. Epoch finished")
    tcpCliSock.send((args.rank + ":" + str(epoch)).encode('utf-8'))

tcpCliSock.close();
stop_thread(t)

