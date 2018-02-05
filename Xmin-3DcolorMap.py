'''Keras Neural Networks Objective Function Regression f(x)=||y-Ax||^2
'''

# import necessary packages
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras import regularizers

from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tr
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow warnings

# number of iterations
epochs = 30

N = 15000  # number of points in the train set
N_test = 3000  # number of points in the test set

# generate train and validation points around minimum [0.0833， 0.833， 0.0833]
X1 = 0.16 * np.random.rand(1, N) - 0.0833
X2 = 1.6 * np.random.rand(1, N) - 0.833
X3 = 0.16 * np.random.rand(1, N) - 0.0833

# generate optimal points to mark the optimal point
X1marker = np.full((1, N),0.0833)
X2marker = np.full((1, N),0.833)
X3marker = np.full((1, N),0.0833)

# concetenate values into a matrix
X = np.concatenate((X1, X2, X3))
Xmarker = np.concatenate((X1marker, X2marker, X3marker))

# get the coordinates for each point
x1 = np.asarray(X[0, :])
x2 = np.asarray(X[1, :])
x3 = np.asarray(X[2, :])

# set Z=A*x for x1 and x2
A = [[4,9,2], [8,1,6], [3,5,7]]

Z = np.dot(A, X)

z1nn = np.asarray(Z[0, :])
z2nn = np.asarray(Z[1, :])
z3nn = np.asarray(Z[2, :])

# set y
y1nn = np.full((1, N),8)
y2nn = np.full((1, N),2)
y3nn = np.full((1, N),5)

# calculate the objective function f(x)=||y-Ax||^2 for each point
fnn = np.transpose((y1nn - z1nn) ** 2 + (y2nn - z2nn) ** 2 + (y3nn - z3nn) ** 2)

# modify X so it can be trainable
X_train = np.squeeze(np.transpose(X))
X_marker = np.squeeze(np.transpose(Xmarker))

# create the model
model = Sequential()

# Network structure
input = Input(shape=(X_train.shape[1],))
hidden1 = Dense(100, activation='elu')(input)
hidden2 = Dense(100, activation='elu')(hidden1)
hidden3 = Dense(100, activation='elu')(hidden2)
hidden4 = Dense(2, activation='elu')(hidden3)
hidden5 = BatchNormalization()(hidden4)

hidden6 = Dense(50, activation='relu')(hidden5)
hidden7 = Dense(50, activation='relu')(hidden6)
hidden8 = Dense(50, activation='relu')(hidden7)
output = Dense(1, activation='relu')(hidden8)

# Encoder
Encoder = Model(input, hidden4)
model = Model(input, output)

# compile the model
model.compile(loss='mean_squared_error',optimizer='rmsprop')

# Early stopping
callback = [EarlyStopping(monitor='val_loss', patience=2, min_delta=0.01 ,verbose=0)]
model.fit(X_train, fnn, epochs=epochs, verbose=1, validation_split=0.25, callbacks=callback)

# generate the test data similarly
X1test = 0.16 * np.random.rand(1, N_test) - 0.0833
X2test = 1.6 * np.random.rand(1, N_test) - 0.833
X3test = 0.16 * np.random.rand(1, N_test) - 0.0833

Xtest = np.concatenate((X1test, X2test, X3test))

x1test = np.asarray(Xtest[0, :])
x2test = np.asarray(Xtest[1, :])
x3test = np.asarray(Xtest[2, :])

Ztest = np.dot(A, Xtest)

z1nntest = np.asarray(Ztest[0, :])
z2nntest = np.asarray(Ztest[1, :])
z3nntest = np.asarray(Ztest[2, :])

y1nntest = np.full((1, N_test),8)
y2nntest = np.full((1, N_test),2)
y3nntest = np.full((1, N_test),5)

# calculate the objective function f(x)=||y-Ax||^2 for each point
fnntest = np.transpose((y1nntest - z1nntest) ** 2 + (y2nntest - z2nntest) ** 2 + (y3nntest - z3nntest) ** 2)

# modify X_test so it can be fed to the network
Xtest = np.squeeze(np.transpose(Xtest))
Test_Predict = np.squeeze(model.predict(Xtest))

Xmarker = np.squeeze(np.transpose(Xmarker))

# predict the objective function for X_test
low_dim=Encoder.predict(Xtest)# low dimensional output
Xmarker_ldim=Encoder.predict(Xmarker)# low dimensional output

#plot the 3d surface
fig = plt.figure()
ax = fig.gca(projection='3d')
surf1 = ax.plot_trisurf(low_dim[:,0],low_dim[:,1], np.squeeze(fnntest), cmap=plt.cm.viridis,
                      linewidth=0, antialiased=False)
plt.title("True Objective")  # set title
fig.colorbar(surf1, shrink=0.5, aspect=5) # Add a color bar which maps values to colors.


fig = plt.figure()
ax = fig.gca(projection='3d')
surf2 = ax.plot_trisurf(low_dim[:,0],low_dim[:,1], Test_Predict, cmap=plt.cm.viridis,
                       linewidth=0, antialiased=False)
plt.title("NN Estimated Objective")
fig.colorbar(surf2, shrink=0.5, aspect=5)# Add a color bar which maps values to colors.
plt.show()  # show plot (this blocks the code: make sure it's at the end of your code)

