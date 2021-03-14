import csv
import numpy as np
import random

def get_input(file_name):
    with open('C:/Users/mahdis/PycharmProjects/CIFP/' + file_name + '.csv') as file:
        reader = csv.reader(file, delimiter=',')
        # y = np.array([])
        # x = np.array([])
        xs = []
        ys = []
        for row in reader:
            temp = [float(i) for i in row[0:len(row) - 1]]
            if temp == []:
                continue
            xs.append(temp)
            ys.append([float(i) for i in row[len(row) - 1:]])
        #shuffle
        c = list(zip(xs, ys))
        random.shuffle(c)
        xs, ys = zip(*c)
        x = np.array(xs)

        # maxx = np.amax(x)
        # maxY = np.amax(y)
        # x = np.divide(x, maxx)
        y = np.array(ys)
        # y = np.divide(y, maxY)
        return x, y
