from torchvision.models.resnet import ResNet, Bottleneck
import torchvision.transforms as transforms
import torchvision
import torch
from torch import nn, optim
from torch.autograd import Variable
import datetime


transform = transforms.Compose(
 [ transforms.ToTensor(),
 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

num_epoches = 15
num_batches = 3
batch_size = 120
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

num_classes = 10
#resnet50 = torchvision.models.resnet50(pretrained=True).cuda()
class NonModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(NonModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.conv1.to('cuda:0')
        self.bn1.to('cuda:0')
        self.relu.to('cuda:0')
        self.maxpool.to('cuda:0')

        self.layer1.to('cuda:0')
        self.layer2.to('cuda:0')

        self.layer3.to('cuda:1')
        self.layer4.to('cuda:1')
        self.avgpool.to('cuda:1')

        self.fc.to('cuda:1')
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x.to('cuda:1'))
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



model_1 = NonModelParallelResNet50().to('cuda:0')
model_2 = ModelParallelResNet50()



def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    total_loss = 0

    #inputs = torch.randn(batch_size, 3, image_w, image_h)

    for i, data in enumerate(train_loader, 0): 
        inputs, labels = data
        #print(labels)
        #print(torch.sparse.torch.eye(num_classes).index_select(0, labels))
        labels = torch.sparse.torch.eye(num_classes).index_select(0, labels)
        inputs, labels = Variable(inputs), Variable(labels)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))

        # run backward pass
        labels = labels.to(outputs.device)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / num_batches
    print('loss: %.4f'%avg_loss)


for epoch in range(num_epoches):
    if epoch == 4:
        # 1 GPU -> 2 GPU
        print('Memory: {memory:.3f} ({cached_memory:.3f})\t'.format(
            memory=(float(torch.cuda.memory_allocated()) / 10**9),
            cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
        start_scaling_time = datetime.datetime.now()
        state = {k: v.clone() for k, v in model_1.state_dict().items()}
        model_2.load_state_dict(state)
        finish_scaling_time = datetime.datetime.now()
        print("Scaling in: {} s".format(finish_scaling_time - start_scaling_time))
        print('Memory: {memory:.3f} ({cached_memory:.3f})\t'.format(
            memory=(float(torch.cuda.memory_allocated()) / 10**9),
            cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
    if epoch == 9:
        # 2 GPU -> 1 GPU
        print('Memory: {memory:.3f} ({cached_memory:.3f})\t'.format(
            memory=(float(torch.cuda.memory_allocated()) / 10**9),
            cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
        start_scaling_time = datetime.datetime.now()
        state = {k: v.clone() for k, v in model_2.state_dict().items()}
        model_1.load_state_dict(state)
        finish_scaling_time = datetime.datetime.now()
        print("Scaling out: {} s".format(finish_scaling_time - start_scaling_time))
        print('Memory: {memory:.3f} ({cached_memory:.3f})\t'.format(
            memory=(float(torch.cuda.memory_allocated()) / 10**9),
            cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
    if epoch < 5:
        train(model_1)
    elif epoch < 10:
        train(model_2)
    else:
        train(model_1)

