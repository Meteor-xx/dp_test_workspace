import torch
from torch import nn
from torch.nn import functional as F

# # 在构造自定义块之前，我们先回顾一下多层感知机 （ 4.3节 ）的代码。 下面的代码生成一个网络，
# # 其中包含一个具有256个单元和ReLU激活函数的全连接隐藏层， 然后是一个具有10个隐藏单元且不带激活函数的全连接输出层。
# net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
#
# net(X)


# 在下面的代码片段中，我们从零开始编写一个块。 它包含一个多层感知机，其具有256个隐藏单元的隐藏层和一个10维输出层。
# 注意，下面的MLP类继承了表示块的类。 我们的实现只需要提供我们自己的构造函数（Python中的__init__函数）和前向传播函数。
# class MLP(nn.Module):
#     # 用模型参数声明层。这里，我们声明两个全连接的层
#     def __init__(self):
#         # 调用MLP的父类Module的构造函数来执行必要的初始化。
#         # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
#         super().__init__()
#         self.hidden = nn.Linear(20, 256)  # 隐藏层
#         self.out = nn.Linear(256, 10)  # 输出层
#
#     # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
#     def forward(self, X):
#         # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
#         return self.out(F.relu(self.hidden(X)))
#
#
# net = MLP()
# net(X)


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
