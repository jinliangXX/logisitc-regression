import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset


def sigmoid(x):
    '''
    实现sigmoid函数
    :param x:
    :return:
    '''
    a = 1 / (1 + np.exp(-x))
    return a


def process_data():
    # 加载数据集
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    # 分别获取训练集数量、测试集数量、训练、测试集里面的图片的宽度和高度（均为64x64）
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    # 把维度为（64，64，3）的numpy数组重新构造为（64 x 64 x 3，1）的数组
    train_set_x_flatten = train_set_x_orig.reshape(
        train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(
        test_set_x_orig.shape[0], -1).T
    # 数据预处理，进行居中和标准化（归一化）
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    return train_set_x, test_set_x, train_set_y, test_set_y


def initialize_with_zeros(dim):
    '''
    初始化参数
    :param dim:
    :return:
    '''
    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    '''
    正向、反向计算
    :param w:
    :param b:
    :param X:
    :param Y:
    :return:
    '''
    m = X.shape[1]

    # 正向计算
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(1 / m) * np.sum(
        np.multiply(Y, np.log(A)) + np.multiply(1 - Y,
                                                np.log(
                                                    1 - A)))
    # 反向传播
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    # 断言检测程序是否错误
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    # 保存相关参数
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate,
             print_cost=False):
    '''
    使用梯度下降更新参数
    :param w:
    :param b:
    :param X:
    :param Y:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    '''
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    '''
    使用模型预测
    :param w:
    :param b:
    :param X:
    :return:
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = 1 / (1 + np.exp(-(np.dot(w.T, X) + b)))

    for i in range(A.shape[1]):

        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def plt_cost(d):
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()


if __name__ == '__main__':
    # 设置超参数
    num_iterations = 2000
    learning_rate = 0.005
    print_cost = True
    # 加载数据集，并进行初步处理
    X_train, X_test, Y_train, Y_test = process_data()

    # 初始化参数
    w, b = initialize_with_zeros(X_train.shape[0])

    # 训练模型，即利用梯度下降更新参数
    params, grads, costs = optimize(w, b, X_train, Y_train,
                                    num_iterations=num_iterations,
                                    learning_rate=learning_rate,
                                    print_cost=print_cost)

    w = params['w']
    b = params['b']

    # 利用训练好的参数进行预测
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # 打印预测结果与准确率
    print("train accuracy: {} %".format(100 - np.mean(
        np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(
        np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    # 描绘成本函数变化图
    plt_cost(d)
