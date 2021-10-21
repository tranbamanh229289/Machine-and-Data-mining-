import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

np.random.seed(1234)
data=pd.read_csv("data_iris.csv").values
np.random.shuffle(data)
def Scaler(X):
    scale=preprocessing.StandardScaler()
    scale=scale.fit(X)
    return scale ,scale.transform(X)
def inverse(scale,X):
    return scale.inverse_transform(X)

def process (ratio):
    X = data[:,:-1]
    Y = data[:,-1:]
    N = data.shape[0]
    index = int(N * ratio)
    Y[np.where(Y == "Setosa")] = 0
    Y[np.where(Y == "Versicolor")] = 1
    Y[np.where(Y == "Virginica")] = 2
    Y=np.array(Y,dtype=np.int32)
    X_train = X[:index, :]
    Y_train = Y[:index,-1]
    X_test = X[index:, :]
    Y_test = Y[index:,-1]
    scale_train,X_train=Scaler(X_train)
    scale_test ,X_test=Scaler(X_test)
    return X_train, Y_train, X_test, Y_test ,scale_train,scale_test

def onehot(Y):
    c=np.unique(Y).shape[0]
    return np.eye(c)[Y]

def graph_accuracy(X_test, Y_test, Y_predict):
    colormap = np.array(['r', 'g', 'b'])
    index_predict = np.array(np.sum([[0, 1, 2]] * Y_predict, axis=1), dtype=np.int32)
    index_test = np.array(np.sum([[0, 1, 2]] * Y_test, axis=1), dtype=np.int32)
    plt.style.use('classic')

    plt.subplot(1, 2, 1)
    plt.title("Predict", size=20)
    plt.xlabel('$length sepal$', size=20)
    plt.ylabel('$width sepal$', size=20)
    plt.scatter(X_test[:, 0], X_test[:, 1], s=100, c=colormap[index_predict])

    plt.subplot(1, 2, 2)
    plt.title("Real", size=20)
    plt.xlabel('$length sepal$', size=20)
    plt.ylabel('$width sepal$', size=20)
    plt.scatter(X_test[:, 0], X_test[:, 1], s=100, c=colormap[index_test])
    plt.show()










