import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import Common

kernel=['linear','rbf']
RATIO = 0.8
C=1;
X_train, Y_train, X_test, Y_test,scale_train,scale_test = Common.process(RATIO)

def svm(X_train,Y_train,X_test,C):
    ls={}
    weight={}
    bias={}
    for item in kernel:
        model = SVC(C=C,kernel=item)
        model.fit(X_train, Y_train)
        Y_predict = model.predict(X_test)
        ls[item]=Y_predict
        if item == "linear":
            weight[item]=model.coef_
            bias[item]=model.intercept_

    return ls ,weight ,bias
def accuracy(Y_test,Y_predict):
    result = np.abs(Y_test - Y_predict)
    count = 0
    for i in result:
        if (i != 0):
            count = count + 1
    N = Y_test.shape[0]
    acc = (N - count) / N
    return acc

ls,weight ,bias=svm(X_train,Y_train,X_test,C)
predict_linear=np.array(ls['linear'])
predict_rbf=np.array(ls['rbf'])
acc_linear =accuracy(Y_test,predict_linear)
acc_rbf=accuracy(Y_test,predict_rbf)
predict_linear=Common.onehot(predict_linear)
predict_rbf =Common.onehot(predict_rbf)
X_train=Common.inverse(scale_train,X_train)
X_test=Common.inverse(scale_test,X_test)
Y_train = Common.onehot(Y_train)
Y_test = Common.onehot(Y_test)
print("Accuracy linear :")
print(acc_linear * 100, "%")
print ("Accuracy rbf :")
print (acc_rbf*100 ,"%")
weight=weight['linear']
bias=bias['linear']

Common.graph_accuracy(X_test,Y_test,predict_linear)
Common.graph_accuracy(X_test,Y_test,predict_rbf)




