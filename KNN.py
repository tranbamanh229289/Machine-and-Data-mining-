import numpy as np
import Common
import matplotlib.pyplot as plt
import pandas as pd

RATIO=0.8
K=9
X_train,Y_train,X_test,Y_test ,scale_train,scale_test =Common.process(RATIO)

def evaluate(X_train,Y_train,X_test,K) :
    ls=[]
    for item in X_test :
        euclid=np.sqrt(np.array(np.sum((X_train-item)**2,axis=1),dtype=np.float16))
        index_sort=euclid.argsort()[0:K]
        dis_neigh=Y_train[index_sort]
        a=np.bincount(dis_neigh).argmax()
        ls.append(a)
    return ls

ls = evaluate(X_train,Y_train,X_test,K)

def accuracy(Y_test,Y_predict):
    result = np.abs(Y_test - Y_predict)
    count = 0
    for i in result:
        if (i != 0):
            count = count + 1
    N = Y_test.shape[0]
    acc = (N - count) / N
    return acc

Y_predict=np.array(evaluate(X_train,Y_train,X_test,K))
acc= accuracy(Y_test,Y_predict)
Y_predict=Common.onehot(Y_predict)
Y_train = Common.onehot(Y_train)
Y_test = Common.onehot(Y_test)
X_train=Common.inverse(scale_train,X_train)
X_test=Common.inverse(scale_test,X_test)
Common.graph_accuracy(X_test, Y_test, Y_predict)
print("Accuracy :")
print(acc * 100, "%")










