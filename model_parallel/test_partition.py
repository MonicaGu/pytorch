# coding=utf-8
from torchvision.models.resnet import ResNet, Bottleneck
import torchvision.transforms as transforms
import torchvision
import torch
from torch import nn, optim
from torch.autograd import Variable
import time

from models import ResNet50
from runtime import runtime


transform = transforms.Compose([ transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

num_classes = 10
num_epoches = 7
num_batches = 3
batch_size = 1200
image_w = 128
image_h = 128

train_data = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=True,
 download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
 shuffle=True)
test_data = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=False,
 download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
 shuffle=False)

loss_fn = nn.MSELoss()
model = ResNet50.model(loss_fn)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, r, optimizer, epoch):
    
    losses = AverageMeter()

    epoch_start_time = time.time()

    for i, data in enumerate(train_loader, 0): 
        inputs, labels = data

        # perform forward pass
        loss = r.run_forward(inputs, labels)
        losses.update(loss)

        # perform backward pass
        optimizer.zero_grad()
        r.run_backward(loss)
        optimizer.step()


    print("Epoch %d: %.3f seconds, loss: %.3f" % (epoch, time.time() - epoch_start_time, losses.avg))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))


# 初始化runtime
r = runtime.StageRuntime(model=model, config='./models/ResNet50/mp_conf_1gpu.json')
print('Memory: {memory:.3f} ({cached_memory:.3f})\t'.format(
    memory=(float(torch.cuda.memory_allocated()) / 10**9),
    cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))

optimizer = torch.optim.SGD(r.parameters(),lr=0.001)
for epoch in range(num_epoches):
    if epoch == 2:
        scale_start_time = time.time()
        #r = runtime.StageRuntime(model=model, config='./models/ResNet50/mp_conf_2gpu.json')
        r.scale(config='./models/ResNet50/mp_conf_2gpu.json')
        scale_end_time = time.time()
        print("\nScaling in: %.3fs" % (scale_end_time - scale_start_time))
        print('Memory: {memory:.3f} ({cached_memory:.3f})\n'.format(
            memory=(float(torch.cuda.memory_allocated()) / 10**9),
            cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
    if epoch == 3:
        scale_start_time = time.time()
        #r = runtime.StageRuntime(model=model, config='./models/ResNet50/mp_conf_1gpu.json')
        r.scale(config='./models/ResNet50/mp_conf_1gpu.json')
        scale_end_time = time.time()
        print("\nScaling out: %.3fs" % (scale_end_time - scale_start_time))
        print('Memory: {memory:.3f} ({cached_memory:.3f})\n'.format(
            memory=(float(torch.cuda.memory_allocated()) / 10**9),
            cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
    if epoch == 5:
        scale_start_time = time.time()
        #r = runtime.StageRuntime(model=model, config='./models/ResNet50/mp_conf_2gpu.json')
        r.scale(config='./models/ResNet50/mp_conf_2gpu.json')
        scale_end_time = time.time()
        print("\nScaling in: %.3fs" % (scale_end_time - scale_start_time))
        print('Memory: {memory:.3f} ({cached_memory:.3f})\n'.format(
            memory=(float(torch.cuda.memory_allocated()) / 10**9),
            cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))

    
    train(train_loader, r, optimizer, epoch)

