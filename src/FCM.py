import random

import numpy as np
import math

from numpy.linalg import linalg


class FCM:

    def __init__(self, m, x, y):
        self.m = m
        self.v = None
        self.x = x
        self.y = y
        self.n_data = np.array(x).shape[0]
        self.xU = None
        self.U = np.zeros(shape=(self.n_data, self.m))
        self.n = self.x.shape[1]  # dimension of data

    def initialize_vs(self):
        # self.v = np.array([[random.uniform(0, 1) for _ in range(2)] for _ in range(self.m)])
        indexes = random.sample(range(0, self.n_data), self.m)
        self.v = np.array([self.x[i] for i in indexes])
        print(self.v)

    def set_v(self, vis):
        self.v = vis

    def update_U(self):
        # self.xU = self.U
        self.xU = np.empty_like(self.U)
        self.xU[:] = self.U
        m = 2
        # m = self.m
        power = float(2 / (m - 1))
        for i in range(self.n_data):
            for j in range(self.m):
                sum = 0
                nd = np.subtract(self.x[i], self.v[j])
                numerator = linalg.norm(nd)
                # numerator = math.sqrt(np.sum(nd ** 2))
                if numerator == 0:
                    numerator = 0.0000000001
                for k in range(self.m):
                    dd = np.subtract(self.x[i], self.v[k])
                    denominator = linalg.norm(dd)
                    # denominator = math.sqrt(np.sum(dd ** 2))
                    if denominator == 0:
                        denominator = 0.0000000001
                    tmp = pow(numerator / denominator, power)
                    sum += tmp
                self.U[i, j] = 1 / sum


    def update_v(self):
        for j in range(self.m):
            numerator, denominator = 0, 0
            for i in range(self.n_data):
                numerator += np.multiply(self.x[i], self.U[i, j]**2)
                denominator += self.U[i, j]**2
            self.v[j] = np.divide(numerator, denominator)



    def terminate(self, sigma):
        difference = np.subtract(self.xU, self.U)
        sum_square = linalg.norm(difference )
        # sum_square = math.sqrt(np.sum(difference ** 2))
        # print("sum_square   " + str(sum_square))
        # print("difference   " + str(np.amax(difference)))
        return sum_square < sigma

    def runFCM(self):
        self.initialize_vs()
        self.update_U()
        counter = 0
        while (not self.terminate(1)):
            counter+=1
            print(counter)
            self.update_v()
            self.update_U()
            if np.array_equal(self.U, self.xU):
                print("equaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaal")
        # print(self.v)
        # print(np.amax(self.U))
