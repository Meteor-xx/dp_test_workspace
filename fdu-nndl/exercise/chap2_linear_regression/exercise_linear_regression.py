import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(map(float, line.strip().split()))
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)


def identity_basis(x):
    ret = np.expand_dims(x, axis=1)
    return ret


def multinomial_basis(x, feature_num=10):
    '''多项式基函数'''
    x = np.expand_dims(x, axis=1)  # shape(N, 1)
    feat = [x]
    for i in range(2, feature_num + 1):
        feat.append(x ** i)
    ret = np.concatenate(feat, axis=1)
    return ret


def gaussian_basis(x, feature_num=10):
    '''高斯基函数'''
    centers = np.linspace(0, 25, feature_num)
    width = 1.0 * (centers[1] - centers[0])
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x] * feature_num, axis=1)

    out = (x - centers) / width
    ret = np.exp(-0.5 * out ** 2)
    return ret


def main(x_train, y_train):
    """
    训练模型，并返回从x到y的映射。

    """
    basis_func = identity_basis
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)  # 建立与x_train维度相同的全为1的序列
    phi1 = basis_func(x_train) #
    phi = np.concatenate([phi0, phi1], axis=1)

    # ==========
    # todo '''计算出一个优化后的w，请分别使用最小二乘法以及梯度下降两种办法优化w'''
    # ==========
    # 最小二乘法
    w = np.dot(np.linalg.pinv(phi), y_train)  # np.linalg.pinv(phi) 求伪逆

    # 梯度下降
    # dim = phi.shape[1]
    # w = 0.05 * np.random.randn(dim)
    # for i in range(10000):
    #     delta_w = np.dot(phi.T, y_train - np.dot(phi, w))
    #     w += 0.001*(delta_w/np.mean(np.abs(delta_w)))

    def f(x):
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_func(x)
        phi = np.concatenate([phi0, phi1], axis=1)
        y = np.dot(phi, w)
        return y
        pass

    return f


def evaluate(ys, ys_pred):
    """评估模型。"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std


# 程序主入口（建议不要改动以下函数的接口）
if __name__ == '__main__':
    train_file = 'train.txt'
    test_file = 'test.txt'
    # 载入数据
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    print(x_train.shape)
    print(x_test.shape)

    # 使用线性回归训练模型，返回一个函数f()使得y = f(x)
    f = main(x_train, y_train)

    y_train_pred = f(x_train)
    std = evaluate(y_train, y_train_pred)
    print('训练集预测值与真实值的标准差：{:.1f}'.format(std))

    # 计算预测的输出值
    y_test_pred = f(x_test)
    # 使用测试集评估模型
    std = evaluate(y_test, y_test_pred)
    print('预测值与真实值的标准差：{:.1f}'.format(std))

    # 显示结果
    plt.plot(x_train, y_train, 'ro', markersize=3)
    #     plt.plot(x_test, y_test, 'k')
    plt.plot(x_test, y_test_pred, 'k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend(['train', 'test', 'pred'])
    plt.show()