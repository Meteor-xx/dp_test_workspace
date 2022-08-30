import os
import pandas as pd
import torch

# 2.2.1. 读取数据集

# 举一个例子，我们首先创建一个人工数据集，并存储在CSV（逗号分隔值）文件 ../data/house_tiny.csv中。
# 以其他格式存储的数据也可以通过类似的方式进行处理。 下面我们将数据集按行写入CSV文件中。
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 要从创建的CSV文件中加载原始数据集，我们导入pandas包并调用read_csv函数。该数据集有四行三列。
# 其中每行描述了房间数量（“NumRooms”）、巷子类型（“Alley”）和房屋价格（“Price”）。
data = pd.read_csv(data_file)
print(data)

# 2.2.2. 处理缺失值

# 注意，“NaN”项代表缺失值。 为了处理缺失的数据，典型的方法包括插值法和删除法， 其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。
# 在这里，我们将考虑插值法。
#
# 通过位置索引iloc，我们将data分成inputs和outputs， 其中前者为data的前两列，而后者为data的最后一列。
# 对于inputs中缺少的数值，我们用同一列的均值替换“NaN”项。
inputs, mid_puts, outputs = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, -1]

inputs = inputs.fillna(inputs.mean())
print(inputs)
# 0    3.0
# 1    2.0
# 2    4.0
# 3    3.0
# Name: NumRooms, dtype: float64

# 对于inputs中的类别值或离散值，我们将“NaN”视为一个类别。
# 由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”， pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。
# 巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。
# 缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。
mid_puts = pd.get_dummies(mid_puts, dummy_na=True)
print(mid_puts)
#    Pave  NaN
# 0     1    0
# 1     0    1
# 2     0    1
# 3     0    1

# 2.2.3. 转换为张量格式
x, y, z = torch.tensor(inputs.values, dtype=int), torch.tensor(mid_puts.values, dtype=int), torch.tensor(outputs.values)
print(x, y, z)
# tensor([3, 2, 4, 3])
# tensor([[1, 0],
#         [0, 1],
#         [0, 1],
#         [0, 1]])
# tensor([127500, 106000, 178100, 140000])
# x原为1行4列，改为4行一列与y按列方向拼接
print(torch.cat((x.reshape(4, 1), y), dim=1))
# tensor([[3, 1, 0],
#         [2, 0, 1],
#         [4, 0, 1],
#         [3, 0, 1]])


