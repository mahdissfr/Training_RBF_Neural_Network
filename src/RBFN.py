import numpy as np
import math


class RBFN:

    def __init__(self, m, x, y, centers, U):
        self.m = m
        self.v = centers
        self.U = U
        self.weights = None
        self.prediction = np.array([])
        self.yhat = None
        self.x = x
        self.y = y
        self.classes = np.unique(self.y)
        self.n_data = np.array(x).shape[0]
        self.G = np.zeros(shape=(self.n_data, self.m))
        self.n = np.array(x).shape[1]  # dimension of data
        self.C = []
        self.YY = np.zeros(shape=(self.n_data, np.array(self.classes).shape[0]))

    def set_YY(self):
        for i in range(np.array(self.y).shape[0]):
            j = np.where(self.classes == self.y[i][0])[0][0]
            self.YY[i, j] = 1

    def set_C(self):
        for i in range(self.m):
            denominator = 0
            numerator = np.zeros(shape=(self.n, self.n))
            for k in range(self.n_data):
                difference = np.subtract(self.x[k], self.v[i])
                transpose = np.transpose(difference)
                mull = np.matmul(difference, transpose)
                numerator = np.add(np.multiply(mull, self.U[k, i] ** 2), numerator)
                denominator += self.U[k, i] ** 2
            ci = np.divide(numerator, denominator)
            self.C.append(np.linalg.pinv(ci))


    def get_accuracy(self):
        difference = np.subtract(self.y, self.yhat)
        siggn = np.sign(difference)
        abbs = abs(siggn)
        return 1 - (np.sum(abbs, axis=0) / self.n_data)

    def set_G(self):
        for j in range(0, self.n_data):
            for i in range(0, self.m):
                difference = np.subtract(self.x[j], self.v[i])
                transpose = np.transpose(difference)
                diffXc_inv = np.matmul(transpose, self.C[i])
                mull = np.matmul(diffXc_inv, difference)
                pwr = np.multiply(mull, -0.1)
                self.G[j, i] = math.exp(pwr)

    def set_W(self):
        gtg = np.matmul(np.transpose(self.G), self.G)
        gty = np.matmul(np.transpose(self.G), self.YY)
        self.weights = np.matmul(np.linalg.pinv(gtg), gty)

    def predict(self):
        self.prediction = np.matmul(self.G, self.weights)

    def set_yhat(self):
        tmp = np.reshape(self.prediction.argmax(axis=1), (len(self.y), 1))
        self.yhat = np.reshape([self.classes[tmp[i][0]] for i in range(len(tmp))], (len(self.y), 1))


    def runRBF_4train(self):
        self.set_YY()
        self.set_C()
        self.set_G()  # G
        self.set_W()  # w
        self.predict()  # GW
        self.set_yhat()
        acc = self.get_accuracy()
        print("accuracy of train: " + str(acc))

    def set_W_from_train(self, w):
        self.weights = w

    def runRBF_4test(self, w):
        self.set_YY()
        self.set_C()
        self.set_G()  # G
        self.set_W_from_train(w)
        self.predict()  # GW
        self.set_yhat()
        acc = self.get_accuracy()
        print("accuracy of test: " + str(acc))
