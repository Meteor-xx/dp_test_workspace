import torch
from torch import nn

print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))
print(torch.cuda.device_count())


def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


try_gpu(), try_gpu(10), try_all_gpus()

x = torch.tensor([1, 2, 3])
print(x.device)

X = torch.ones(2, 3, device=try_gpu())
print(X)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(try_gpu())
print(net(X))
