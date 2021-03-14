import math
import random

from matplotlib.pyplot import figure, show
from matplotlib import pyplot as plt, pyplot
import numpy as np
from numpy.linalg import linalg


def categorize_result(x_test, y_test, yhat, classes, vis):
    clr = ["blue","green","cyan","magenta","yellow"]
    indexes =random.sample(range(0, len(clr)), len(classes))
    clsclr = [clr[i] for i in indexes]
    clsclr.extend(["red","black"])
    print(clsclr)
    xs = []
    ys = []
    label = classes.tolist()
    for k in range(len(clsclr)):
        xs.append([])
        ys.append([])
    for i in range(len(x_test)):
        if y_test[i] != yhat[i]:
            j = clsclr.index("red")
        else:
            j = label.index(y_test[i])
        xs[j].append([x_test[i, 0]])
        ys[j].append([x_test[i, 1]])
    for i in range(len(vis)):
        j = clsclr.index("black")
        xs[j].append([vis[i, 0]])
        ys[j].append([vis[i, 1]])
    return xs, ys, clsclr



def plot(x_test, y_test, yhat, classes, vis):
    xs, ys, clsclr = categorize_result(x_test, y_test, yhat, classes, vis)
    fig = figure()
    axs = []
    for i in range(len(clsclr)):
        axs.append(fig.add_subplot(111, label="1"))
        axs[i].scatter(xs[i], ys[i], label="0", color=clsclr[i], marker="o")
    show()

def plot_boundaries(centers):
    x_centers = []
    y_centers = []
    for i in range(0, len(centers)):
        x_centers.append(centers[i, 0])
        y_centers.append(centers[i, 1])
    pyplot.xlim()
    pyplot.ylim()
    x_range = np.arange(-5, 15, 0.1)
    y_range = np.arange(-5, 15, 0.1)
    xx, yy = np.meshgrid(x_range, y_range)
    cmap = pyplot.get_cmap('Paired')
    zz = np.zeros(xx.shape)
    for i in range(zz.shape[0]):
        for j in range(zz.shape[1]):
            uik = []
            x_vector = [xx[i][j], yy[i][j]]
            for ci in centers:
                temp = compute_U(x_vector, ci,centers)
                uik.append(temp)
            zz[i][j] = np.argmax(uik)
    pyplot.pcolormesh(xx, yy, zz, cmap=cmap)
    pyplot.plot(x_centers, y_centers, 'ro', color='black')
    pyplot.show()

def compute_U( x, center, centers):
    m=2
    sum_distance = 0
    for k in range(0, len(center)):
        sum_distance += ((math.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)) / math.sqrt(((x[0] - centers[k][0])**2 + (x[1] - centers[k][1])**2))) ** (2 / (m - 1))
    return 1.0 / sum_distance






