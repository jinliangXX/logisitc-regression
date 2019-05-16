# logisitc-regression
logistic回归（逻辑回归、对率回归）代码实现

###文件介绍
- main.py          程序运行入口
- lr_utils.py      数据集加载工具类
- datasets         数据集文件

###需要的工具包
- numpy
- h5py
- matplotlib

###logistic回归讲解
####地址：
---
layout:     post                    # 使用的布局
title:      logistic回归           # 标题 
subtitle:   logistic回归介绍与源码分析 
date:       2019-04-26              # 时间
author:     Jinliang                      # 作者
header-img: img/post-bg-os-metro.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
mathjax: true                       #是否显示公式
tags:                               #标签
    - Machine Learning
    - Deep Learning

---

## 1. 介绍（由线性模型引出logistic回归）

首先介绍一下什么是线性模型呢？

线性模型的定义如下：给定$d$个属性描述的样本$\textbf x=(x_1,x_2,\ldots,x_d)$，$x_i$代表样本在第$i$个属性上的取值。

线性模型的目的是学习一个函数，它可以通过属性的线性组合来进行预测。


$$
f(\textbf x)=w_1x_1+w_2x_2+\ldots+w_dx_d+b\\
向量形式为：f(\textbf x)=\textbf w^T\textbf x+b
$$


线性模型中的$\textbf x$直观的表达了各个属性在预测中的重要性，具有很好的可解释性，一些非线性模型可在线性模型的基础上引入**层级结构**或**高维映射**得到。

------

线性模型可解决回归任务和分类任务，让我们先来看回归任务。

1. 回归任务

   - [简单线性回归：一个样本仅有一个属性]: https://jinliangxx.github.io/2019/04/03/ML-100Days-Day2

   - [多元线性回归：一个样本有多个属性]: https://jinliangxx.github.io/2019/04/04/ML-100Days-Day3

2. 分类任务

   logistic回归，有时也被称为**逻辑回归**，但是部分人认为逻辑回归不准确，应该译为**对数几率回归**或**对率回归**，在此处，我们不多做解释，仅称其为logistic回归。

   logistic到底是什么呢？怎么来的？

## 2. logistic回归的来历

> 我们可以通过简单线性回归、复杂线性回归等线性模型完成回归任务，怎么讲线性模型应用于分类任务中呢，尤其是二分类。(提示：可以参考*对数线性回归*的原理)

答案：需要找到一个单调可微函数，将线性模型的输出和分类任务的真实标记$y$联系起来。

接下来，我们需要考虑这样的单调可微函数是什么。

分析：线性模型的输出范围是$(-\infty,+\infty)$，二分类任务的真实标记值为$\{0，1\}$，因此，显而易见，我们需要一个函数，能够将$(-\infty,+\infty)$转化为$\{0，1\}$，这种理想的函数应该是'*单位阶跃函数*'，单位阶跃函数到底啥样子呢，我们通过公式和函数图像了解一下。

公式：
$$
y=
\begin{cases}
0,&z<0;\\
0.5,&z=0;\\
1,&z>0,
\end{cases}
$$
公式的含义是当$z$大于0时，判定为正例；小于0判定为反例；等于0时，可以判别为正例或者反例。

函数图像如下：

![image-20190505230100796](https://jinliangxx.oss-cn-beijing.aliyuncs.com/2019-05-05-150621.jpg)

图3.2（来自西瓜书）中的红色线代表的就是单位阶跃函数，但是单调阶跃函数不连续，因此不能直接用于将回归任务变为二分类任务。

我们希望找到一个替代函数，他要求有以下性质：

1. 单调可微
2. 连续
3. 近似单位阶跃函数

次数引入一个新的概念：**Sigmoid函数**

> 形似S的函数，logistic函数是其中最重要的代表。

因此logistic函数可以作为单位阶跃函数的替代函数：
$$
y=\frac{1}{1+e^{-z}}
$$
函数的图像如图3.2中黑色线所示，实际上可以理解为对数几率函数将$z$值(线性回归的预测值)转换为接近0、1的值，输出值在$z=0$处变化很大很陡，将$z$的展开带入上式得：
$$
y=\frac{1}{1+e^{-(\textbf w^T\textbf x+b)}}，可变化为\ln \frac{y}{1-y}=\textbf w^T\textbf x+b
$$
公式4实际上将线性回归的真实结果去逼近真实标记的对数几率，因此称其为**logistic回归**。

## 3. 优点

logistic回归方法有很多优点：

1. 直接对分类可能性进行建模，**无需事先假设概率分布**，避免了假设分布不准确带来的问题。
2. 它预测的不是类别，而是可得到**近似概率预测**，对许多利用概率辅助决策的任务很有用。
3. logistic回归求解的目标函数是任意阶可导的凸函数，有很好的数学性质，许多数值优化算法可以直接用于求取最优解。



## 4. logistic的损失函数(Loss Function)和成本函数(Cost Function)

首先介绍一下损失函数和成本函数的概念：

**损失函数：**即Loss Function，也叫误差函数，用来衡量单个样本预测输出值和实际输出值有多接近，用$L(\hat y,y)$表示。损失函数最常用的方法是**均方误差**，即预测值和实际值的平方差。

**成本函数：**即Cost Function，也叫代价函数，用来衡量整体样本参数的总代价，用$J(W,b)$表示。

**两者之间的关系：**损失函数针对单个样本而言，成本函数针对整体样本而言。成本函数相当于求取整体样本的平均损失，因此可以表示为$J(W,b)=\frac{1}{m} \sum_{i=1}^mL(\hat y,y)$。

ps：这里的m不一定值全部的样本数量，意义有点类似于单次更新使用样本的数量，例如在mini-batch梯度下降中，这里的m代表mini-batch的大小。([mibi-batch梯度下降](https://jinliangxx.github.io/2019/05/01/%E5%B8%B8%E8%A7%81%E7%9A%84%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/#3-mini-batch%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D)的介绍再另一篇博文中)

------

介绍完损失函数和成本函数的相关概念，自然而然的问题就是logistic回归的损失函数和成本函数是什么，我们在这里直接给出，然后再进行分析：


$$
L(\hat y,y)=-y\log \hat y-(1-y)\log(1-\hat y)\\
J(W,b)=\frac{1}{m} \sum_{i=1}^mL(\hat y,y)=\frac{1}{m} \sum_{i=1}^m-y\log \hat y-(1-y)\log(1-\hat y)
$$


以上分别是logistic回归的损失函数和成本函数。

咦？是不是很奇怪，logistic回归那么简单，为何不使用均方误差，而采用那么复杂的损失函数呢？

原因是在我们使用优化算法学习逻辑回归参数时，要保证我们的优化目标，即损失函数是个**凸函数**（[凸函数](https://jinliangxx.github.io/2019/05/01/%E5%B8%B8%E8%A7%81%E7%9A%84%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/#1-batch%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95%E7%AE%80%E7%A7%B0%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95)的介绍在另一篇博文中），但是在logistic回归中使用均方误差会导致损失函数不是凸函数，优化算法只能找到多个局部最优解，找不到全局最优解，因此不使用均方误差作为logistic回归的损失函数。

还有一个问题，为何使用上述的公式作为损失函数呢？

额······我尽量尝试解答，原谅我个数学渣渣~~~(以下部分可忽略)

------

在本文的第二部分，我们得出了求取$\hat y$的表达式，即$\hat y=\frac{1}{1+e^{-(\textbf w^T\textbf x+b)}}$。

在此，我们约定$\hat y$表示给定训练样本$x$条件下$y$等于1的概率，即$\hat y=p(y=1|x)$；因为logistic回归解决的是二分类任务，因此$1-\hat y$代表的就是$y$等于0的概率。

整理成公式如下：


$$
y=1 : p(y|x)=\hat y\\
y=0 : p(y|x)=1-\hat y\\
$$


因为logictic回归解决的是二分类问题，因此我们可以将上述公式合并成一个公式：


$$
p(y|x)=\hat y ^y(1-\hat y)^{(1-y)}
$$


别问我为啥这么合并，但是不知道这么合并的原因，我们可以验证这么合并是否准确！

- 当y=0时：

  $p(y|x)=\hat y ^y(1-\hat y)^{(1-y)}=1-\hat y$

  与合并之前的第二个公式一样，木有问题！

- 当y=1时：

   $p(y|x)=\hat y ^y(1-\hat y)^{(1-y)}=\hat y$

  与合并之前的第一个公式一样，木有问题！

结论：按照上述方式合并木有问题！！！

我们继续~

我们的目标是最大化条件概率，为何要最大化条件概率，我的理解是对于$p(y|x)$，它的值越大，证明它越接近真实值，我们的预测越准确。

因为$\log$函数是严格的单调递增函数，因此最大化$\log(p(y|x))$相当于最大化$p(y|x)$，我们对$\log(p(y|x))$进行化简：


$$
\log(p(y|x))=y\log \hat y+(1-y)\log(1-\hat y)
$$


咦！是不是特别熟悉，有点像之前我们给出的损失函数了。

最有一步就是从最大化条件概率和最小化损失函数（原因是优化算法通过凸函数寻找最优解的过程）的转换，我们现在已知最大化的条件概率，要求解损失函数，加个负号(-)就可以了。

因此，得到最终的损失函数：$L(\hat y,y)=-y\log \hat y-(1-y)\log(1-\hat y)$

得到损失函数，成本函数可以通过损失函数求得：$J(W,b)=\frac{1}{m} \sum_{i=1}^mL(\hat y,y)$



## 5. batch梯度下降法

根据数据量的大小和模型的难易程度，我们从batch梯度下降、mini-batch梯度下降、随机梯度下降中选择batch梯度下降法。

因为模型较为简单，不使用如Momentum、RMSprop、Adam类似的用于加速训练的复杂优化算法。

以上优化算法的具体介绍和如何选择的经验请参照另一篇博文：[常见的优化算法](https://jinliangxx.github.io/2019/05/01/%E5%B8%B8%E8%A7%81%E7%9A%84%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/)

在此不过多赘述~



## 6. logistic回归整体的公式推导

为了对logistic加深理解，且logistic模型及运算流程比较简单，我们不使用tensorflow、pytorch等机器学习框架，因此除了logistic的正向计算，其反向传播(BP)我们也进行手动推导及实现。

我们先进行logisitc的**正向计算**：


$$
\textbf z=\textbf w^Tx+b\\
\hat y=sigmoid(\textbf z)=\frac{1}{1+e^{-\textbf z}}=\frac{1}{1+e^{-(\textbf w^T\textbf x+b)}}
$$


正向计算比较简单，就是先进行多元线性回归，然后使用sigmoid函数对线性回归结果进行处理。

------

下面通过**链式法则**（这个还是可以会的哈）进行**反向传播**：

因为我们使用batch梯度下降法，因此需要用到成本函数：$J(W,b)==\frac{1}{m} \sum_{i=1}^m-y\log \hat y-(1-y)\log(1-\hat y)$，这里的$m$代表所有样本的数量。

求导公式回顾：

$y=\log_ax$

$y^`=\frac{1}{x\ln a}$

因此，根据链式法则，首先求$\frac{dJ(W,b)}{d\hat y}$：


$$
\begin{align}
\frac{dJ(W,b)}{d\hat y}&=\frac{1}{m}\sum_{i=1}^m-y\frac{1}{\hat y}-(1-y)\frac{1}{1-\hat y}*(-1)\\
&=\frac{1}{m}\sum_{i=1}^m-\frac{y}{\hat y}+\frac{1-y}{1-\hat y}\\
&=\frac{1}{m}\sum_{i=1}^m\frac{1-y}{1-\hat y}-\frac{y}{\hat y}
\end{align}
$$


求解出$\frac{dJ(W,b)}{d\hat y}$，进而求$\frac{dJ(W,b)}{dz}$:
$$
\begin{align}
\frac{dJ(W,b)}{dz}&=\frac{dJ(W,b)}{d\hat y}·\frac{d\hat y}{dz}\\
&=(\frac{1}{m}\sum_{i=1}^m\frac{1-y}{1-\hat y}-\frac{y}{\hat y})·(\hat y(1-\hat y))\\
&=\frac{1}{m}\sum_{i=1}^m\hat y(1-y)-y(1-\hat y)\\
&=\frac{1}{m}\sum_{i=1}^m\hat y-y
\end{align}
$$


哈哈，最终结果是不是好简单，继续哈

求解$\frac{dJ(W,b)}{dW}$：


$$
\begin{align}
\frac{dJ(W,b)}{dW}&=\frac{dJ(W,b)}{dz}·\frac{dz}{dW}\\
&=(\frac{1}{m}\sum_{i=1}^m\hat y-y)·x\\
&=\frac{x}{m}\sum_{i=1}^m\hat y-y
\end{align}
$$


同理，求解$\frac{dJ(W,b)}{db}$：


$$
\begin{align}
\frac{dJ(W,b)}{db}&=\frac{dJ(W,b)}{dz}·\frac{dz}{db}\\
&=(\frac{1}{m}\sum_{i=1}^m\hat y-y)·1\\
&=\frac{1}{m}\sum_{i=1}^m\hat y-y
\end{align}
$$


求得$dW,db$，可以根据梯度下降的公式进行计算，即：$w:=w-\alpha \frac{dJ(w,b)}{dw},b:=b-\alpha \frac{dJ(w,b)}{db}$

以上就是logistic回归的正向与反向推导，其实还是很简单的，反向推导略微涉及一点数学导数知识，但是相信难不住我们。

其实我们可以以一个上帝的角度观察正向与反向推导的过程，发现正向与反向通过成本函数，或者说损失函数连接起来，实际上确实如此，即使在很复杂的神经网络系统中，损失函数的设计也是非常重要的，在后期我们使用tensorflow、pytorch等工具时，只需要进行提供正向的思路，以及损失函数，复杂的反向推导工具可以完全帮助我们实现，不过在学习初期，我们还是手动推倒一下比较好。

在下一部分，我们会使用python语言实现上述的推导过程，在推导过程中我们总是可以看见$\sum$函数，在python预言实现时，我们可以使用向量化的技术巧妙计算，不仅可以减少代码量，还能利用资源，并行执行节省时间。



## 7. 代码实现

**sigmoid函数：**

要使用numpy哈，它是用Python进行科学计算的基本软件包，我们的向量化离不开他，例如下面的函数，不仅可以以单个数作为输入，还可以将向量、矩阵作为输入。

```python
def sigmoid(x):
    '''
    实现sigmoid函数
    :param x:
    :return:
    '''
    a = 1 / (1 + np.exp(-x))
    return a
```

**处理数据集：**

```python
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
```

**初始化参数：**

```python
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
```

**正向、反向计算：**

```python
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
```

**使用梯度下降算法进行更新：**

```python
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
```

**使用训练好的模型进行预测：**

```python
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
```

**绘制图：**

```python
def plt_cost(d):
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()
```

**结果：**

```
Cost after iteration 0: 0.693147
Cost after iteration 100: 0.584508
Cost after iteration 200: 0.466949
Cost after iteration 300: 0.376007
Cost after iteration 400: 0.331463
Cost after iteration 500: 0.303273
Cost after iteration 600: 0.279880
Cost after iteration 700: 0.260042
Cost after iteration 800: 0.242941
Cost after iteration 900: 0.228004
Cost after iteration 1000: 0.214820
Cost after iteration 1100: 0.203078
Cost after iteration 1200: 0.192544
Cost after iteration 1300: 0.183033
Cost after iteration 1400: 0.174399
Cost after iteration 1500: 0.166521
Cost after iteration 1600: 0.159305
Cost after iteration 1700: 0.152667
Cost after iteration 1800: 0.146542
Cost after iteration 1900: 0.140872
train accuracy: 99.04306220095694 %
test accuracy: 70.0 %
```

![image-20190516225842513](https://jinliangxx.oss-cn-beijing.aliyuncs.com/2019-05-16-145843.png)





## 8. 总结

logistic回归严格意义上说并不属于深度学习，因为他仅有一个隐藏层，而深度学习要求隐藏层数量大于等于2

但是logistic回归却是深度学习入门时很好的研究资料，通过后续的学习我们会明白，深度学习实在类似logistic回归的基础上一步步的增加复杂隐藏层的，但是无论怎样变化，万变不离其宗，因此，强烈建议手动推导并实现一次logistic回归。





