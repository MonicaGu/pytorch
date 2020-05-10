# coding=utf-8
import torch
import os
from torch import nn
import torchvision
import datetime

class Model_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,10,3,1,1)
        self.conv2 = nn.Conv2d(10,20,3,2,1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
 
class Model_B(nn.Module):
    def __init__(self,num_class=5):
        super().__init__()
        self.conv1 = nn.Conv2d(20,40,3,2,1)
        self.conv2 = nn.Conv2d(40,10,3,2,1)
        self.adpool = nn.AdaptiveAvgPool2d([1,1])
        self.linear = nn.Linear(10,num_class)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adpool(x) # n*c*1*1
        x = self.linear(x.view(x.size(0),-1)) # needs reshape
        return x
 
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = Model_A().to('cuda:0')
        self.model_b = Model_B().to('cuda:0')
    def forward(self, x):
        x = self.model_a(x.to('cuda:0'))
        x = self.model_b(x.to('cuda:0'))
        return x

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = Model_A().to('cuda:0')
        self.model_b = Model_B().to('cuda:1')
    def forward(self, x):
        x = self.model_a(x.to('cuda:0'))
        x = self.model_b(x.to('cuda:1'))
        return x

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
 
softmax_func = nn.CrossEntropyLoss()
 
batch = 4
num_class = 5
inputs = torch.rand([batch,3,224,224])
labels = torch.randint(0,num_class,[batch,]) 
model1 = Model1()
model1.train()
for k, v in model1.state_dict().items():
    print(k, v)
#model2 = Model2()
#model2.train()

for buf in model1.buffers():
    print(type(buf.data), buf.size())
    print(buf)
 
optimizer1=torch.optim.Adam(model1.parameters(),lr=0.001,betas=(0.9,0.99),weight_decay=0.0005)
#optimizer2=torch.optim.Adam(model2.parameters(),lr=0.001,betas=(0.9,0.99),weight_decay=0.0005)

# 手动定义两个模型
# naive模式：overhead

for i in range(100):
    if i == 50:
        dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        print(dt_ms, "Begin to copy model")
        #copy model parameters
        model1_dict = model1.state_dict()
        #print(dir(model1))
        print("model1.state_dict", model1_dict[0])
        state = {k: v.clone() for k, v in model1.state_dict().items()}
        model2.load_state_dict(state)
        #print("model2.state_dict", model2.state_dict())
        dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        print(dt_ms, "Finished copying model")

        #for each in model1.parameters():
        #    print (each.grad)
        #for each in model2.parameters():
        #    print (each.grad)
    if i < 50:
        optimizer1.zero_grad()
 
        inputs = inputs.cuda(0)
        labels = labels.cuda(0)
 
        out = model1(inputs)
 
        loss = softmax_func(out, labels)
        print('loss: %.4f'%loss.item())
        loss.backward() # (step k) copy model from GPU0 to GPU1
        # (step k) load model on all GPUS
        optimizer1.step()
        break
    else:
        optimizer2.zero_grad()
 
        inputs = inputs.cuda(0)
        labels = labels.cuda(1)
 
        out = model2(inputs)
 
        loss = softmax_func(out, labels)
        print('loss: %.4f'%loss.item())
        loss.backward() # (step k) copy model from GPU0 to GPU1
        # (step k) load model on all GPUS
        optimizer2.step()


