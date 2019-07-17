import numpy as np

import random

import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt

import sklearn.linear_model


random.seed(0)
#创造二分类的数据集
X, y = sklearn.datasets.make_moons(noise=0.2,n_samples=100,random_state=3121)


#可视化数据集
plt.scatter(X[:,0],X[:,1],s=40,c=y)
X=np.concatenate(((np.zeros(len(X))+1).reshape(100,-1),X),axis=1)


#将y改为-1，+1集合
y=y*2-1

#初始化W,随机
W=np.array([3,2,13])


#算法核心
for i in range(1000):
    
    choose=random.choice(np.arange(len(X)))
    if (np.dot(X[choose],W)*y[choose])<0:
      


      W=W+y[choose]*X[choose]

#可视化分类线
line_x=np.arange(-2,2,1)

line_y=(-W[1]*line_x-W[0])/W[2]


plt.plot(line_x,line_y)

plt.show()


