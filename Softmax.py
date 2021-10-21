import Common
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RATIO = 0.8
EPOCHS = 500
LEARN_RATE = 0.01
INDENTIFICATION_RATE = 0.6
# Read training data
X_train, Y_train, X_test, Y_test,scale_train,scale_test = Common.process(RATIO)

def preprocessing (X_train,Y_train ,X_test ,Y_test):
    X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
    X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)
    Y_train = Common.onehot(Y_train)
    Y_test = Common.onehot(Y_test)
    return X_train,Y_train,X_test,Y_test

X_train, Y_train, X_test, Y_test =preprocessing(X_train,Y_train ,X_test ,Y_test)

def softmax(Z):
    Z = np.array(Z, dtype=np.float32)
    return (np.exp(Z) / np.sum(np.exp(Z), axis=1).reshape(Z.shape[0], 1))

# Cross Entropy
def cost(X, Y, W):
    Z = np.array(np.dot(X, W), dtype=np.float32)
    return -np.sum(Y * np.log(softmax(Z)))


def gradient(Y, X, W, learningrate, k):
    loss = []
    for i in range(k):
        Z = np.array(np.dot(X, W), dtype=np.float32)
        delta = np.dot((Y - softmax(Z)).T, X).T
        W = W + learningrate * delta
        loss.append(cost(X, Y, W))
    return W, loss

W = np.zeros((5, 3))
W, loss = gradient(Y_train, X_train, W, LEARN_RATE, EPOCHS)


def accuracy(W, X_test, Y_test, ratio):
    Y_predict = softmax(np.dot(X_test, W))
    Y_predict[np.where(Y_predict > ratio)] = 1
    Y_predict[np.where(Y_predict < ratio)] = 0
    result = np.sum(np.abs(Y_test - Y_predict), axis=1)
    count = 0
    for i in result:
        if (i != 0):
            count = count + 1
    N = Y_test.shape[0]
    acc = (N - count) / N
    return acc, Y_predict

acc, Y_predict = accuracy(W, X_test ,Y_test, INDENTIFICATION_RATE)

def graph_cost(loss, EPOCHS):
    plt.title("Loss", size=20)
    plt.xlabel('$epochs$', size=20)
    plt.ylabel('$error$', size=20)
    plt.plot(np.arange(EPOCHS), loss)
    plt.show()

X_train=Common.inverse(scale_train ,X_train[:,:-1])
X_test=Common.inverse(scale_test,X_test[:,:-1])
graph_cost(loss, EPOCHS)
Common.graph_accuracy(X_test, Y_test, Y_predict)
print("Accuracy :")
print(acc * 100, "%")
