from FCM import FCM
from RBFN import RBFN
from file_handler import get_input
import numpy as np

from plot import plot, plot_boundaries

xs, ys = get_input('4clstrain1200')
m = 4
n_train = int(np.array(xs).shape[0] * 0.70)
xs_train, ys_train = xs[0:n_train], ys[0:n_train]
xs_test, ys_test = xs[n_train + 1:], ys[n_train + 1:]
fcm = FCM(m, xs_train, ys_train)
fcm.runFCM()
train_rbfn = RBFN(m, xs_train, ys_train, fcm.v, fcm.U)
train_rbfn.runRBF_4train()
test_fcm=FCM(m, xs_test, ys_test)
test_fcm.set_v(fcm.v)
test_fcm.update_U()
test_rbfn = RBFN(m, xs_test, ys_test, test_fcm.v, test_fcm.U)
test_rbfn.runRBF_4test(train_rbfn.weights)

plot(xs_test, ys_test, test_rbfn.yhat, test_rbfn.classes, fcm.v)
plot_boundaries(fcm.v)

