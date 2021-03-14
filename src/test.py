# /Documents/University/S9/CI/Project/CI_final project
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from fcm import fcmean, fcmpred

# df = pd.read_csv('2clstrain1200.csv', names=['X', 'Y', 'label'])
# df = pd.read_csv('4clstrain1200.csv', names=['X', 'Y', 'label'])
df = pd.read_excel('5clstrain1500.xlsx', names=['X', 'Y', 'label'])
train, test = train_test_split(df, test_size=0.3)

X_train, Y_train = train.to_numpy()[:, [0, 1]], train['label'].to_numpy()
X_test, Y_test = test.to_numpy()[:, [0, 1]], test['label'].to_numpy()

enc = OrdinalEncoder(dtype=np.int)
Y_train = enc.fit_transform(Y_train.reshape(-1, 1))
Y_test = enc.fit_transform(Y_test.reshape(-1, 1))

MG = 8  # number of clusters (different from m in centroid formula)
gamma = 0.4
# print(X_train.shape) # (#number of train samples, 2)
m = 2
number_of_clustes = MG
iterations = 20


def train(X_train, Y_train):
    centroids, labels, U = fcmean(number_of_clustes, iterations, X_train, m)
    encoder = OneHotEncoder(sparse=False, categories='auto')
    Y = encoder.fit_transform(Y_train.reshape(-1, 1))
    n = Y_train.shape[0]  ##  number of training samples
    d = X_train.shape[1]  ## dimension of data
    num_classes = Y.shape[1]  ## number of classes
    G = np.ndarray(shape=(n, MG))
    C = np.zeros(shape=(MG, d, d))

    plt.scatter(X_train[:, 0], X_train[:, 1], c=np.asarray(labels, dtype=np.int32), alpha=0.9, marker='x')
    for i in range(len(centroids)):
        plt.plot(centroids[i][0], centroids[i][1], 'b*', markersize=10)
    plt.show()

    for i in range(MG):
        sm = 0
        for j in range(n):
            diff = np.array(X_train[j] - centroids[i]).reshape(-1, 1)
            C[i] += (U[j, i] ** m) * diff * (diff.transpose())
            sm += U[j, i] ** m
        C[i] /= sm

    C = np.array([np.linalg.inv(c) for c in C])

    for i in range(MG):
        for j in range(n):
            diff = np.array(X_train[j] - centroids[i]).reshape(-1, 1)
            G[j, i] = np.exp(-gamma * (diff.transpose().dot(C[i])).dot(diff))

    W = np.linalg.inv(G.T.dot(G)).dot(G.T).dot(Y)

    y_pred = np.argmax(G.dot(W), axis=1)

    acc = np.mean(np.equal(Y_train.flatten(), y_pred.flatten()))

    print('Train Accuracy: ', acc)

    return W, centroids


def pred(X, Y, centroids, W):
    n = X.shape[0]  ##  number of training samples
    d = X.shape[1]  ## dimension of data
    G = np.ndarray(shape=(n, MG))
    C = np.zeros(shape=(MG, d, d))

    U = fcmpred(X, centroids, m, number_of_clustes)

    for i in range(MG):
        sm = 0
        for j in range(n):
            diff = np.array(X[j] - centroids[i]).reshape(-1, 1)
            C[i] += (U[j, i] ** m) * diff * (diff.transpose())
            sm += U[j, i] ** m
        C[i] /= sm

    C = np.array([np.linalg.pinv(c) for c in C])

    for i in range(MG):
        for j in range(n):
            diff = np.array(X[j] - centroids[i]).reshape(-1, 1)
            G[j, i] = np.exp(-gamma * (diff.transpose().dot(C[i])).dot(diff))

    y_pred = np.argmax(G.dot(W), axis=1)

    acc = np.mean(np.equal(Y.flatten(), y_pred.flatten()))
    print('Test Accuracy: ', acc)

    return pred


W, centroids = train(X_train, Y_train)
_ = pred(X_test, Y_test, centroids, W)

k = 2
x_min, x_max = X_test.T[0].min() - k, X_test.T[0].max() + k
y_min, y_max = X_test.T[1].min() - k, X_test.T[1].max() + k
h = 0.1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
z = np.c_[xx.ravel(), yy.ravel()]

zpred = fcmpred(z, centroids, m, number_of_clustes)
frontier = np.argmax(zpred, axis=1)

frontier = frontier.reshape(xx.shape)
plt.contourf(xx, yy, frontier, cmap=plt.cm.Spectral)
plt.ylabel('x')
plt.xlabel('y')
plt.scatter(X_test[:, 0], X_test[:, 1], c=np.squeeze(Y_test), alpha=0.9, marker='x')
for i in range(len(centroids)):
    plt.plot(centroids[i][0], centroids[i][1], 'b*', markersize=10)
plt.show()