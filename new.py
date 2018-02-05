
from keras.models import Model
from keras.layers import Dense, Input
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tr
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from mpl_toolkits.mplot3d import Axes3D
import os

# seed of random data generation
np.random.seed(201724)  # 20172435 #2017243
# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# number of iterations
epochs = 30

##########################INPUT DATASET###################################
lim1 = 0.5  # limit the range of values
N = 12000  # number of points in the train set
# N_test=5000# number of points in the test set

# randomly generate points between lim and -lim
a = -lim1
b = lim1

A = [[4, 9, 2],
     [8, 1, 6],
     [3, 5, 7]]

y = [8, 2, 5]

x_optimal = np.dot(np.linalg.inv(A), y)
# generate Y array
Y = np.transpose(y * np.ones((N, 3)))
# generate X optimal array
X_optimal = np.transpose(x_optimal * np.ones((N, 3)))
# generate X array
X = (b - a) * np.random.rand(3, N) + a
# combine X and X optimal arrays
X = X + X_optimal
# generate X products (X1X2, X2X3, X1X3, X1X2X3) arrays
X_product = np.zeros((4, N))
X_product[0, :] = X[0, :] * X[1, :]  # X1X2
X_product[1, :] = X[1, :] * X[2, :]  # X2X3
X_product[2, :] = X[0, :] * X[2, :]  # X1X3
X_product[3, :] = X[0, :] * X[1, :] * X[2, :]  # X1X2X3
# generate combined X (X1, X2, X3, X1**2, X2**2, X3**2, X1X2, X2X3, X1X3, X1X2X3) array
x_train = np.concatenate((X, X ** 2, X_product))
# generate target function ||y-A*X||**2 for each dimension X1, X2, X3
f = (Y - np.dot(A, X)) ** 2
# generate convex projection points for X1, X2, X3
for i in range(len(X)):
    for j in range(len(X[i])):
        # print(X[i][j])
        if X[i][j] < 0:
            X[i][j] = 100000
        else:
            X[i][j] = 0
# generate target function ||y-A*X||**2 + summation(xi) (xi>0)
y_train = sum(f) + sum(X)
# transpose X train
X_train = np.transpose(x_train)

##########################GRADIENT DESCENT###################################
# number of iteration
iter = 50
# learning rate
tau = 0.001
# generate gradient descent starting points arrays
x_gd = np.zeros((3, iter))
x_gd[0, 0] = X_product[0, 0]
x_gd[1, 0] = X_product[1, 0]
x_gd[2, 0] = X_product[2, 0]

# compute the result of each iteration of X(t+1)=X(t)-tau*2*A'*(A*X(t)-y)
for i in range(1, iter - 1):
    x_gd[:, i] = x_gd[:, i - 1] - tau * 2 * np.dot(np.transpose(A), np.dot(A, x_gd[:, i - 1]) - y)
# generate  proximal algorithm for X1, X2, X3
for i in range(len(x_gd)):
    for j in range(len(x_gd[i])):
        if x_gd[i][j] < 0:
            x_gd[i][j] = 0
# generate target function ||y-A*X||**2 for each dimension
f_gd = (Y[:, 0:50] - np.dot(A, x_gd)) ** 2
# generate target function ||y-A*X||**2
y_gd = sum(f_gd)

# generate X gd product (X1X2, X2X3, X1X3, X1X2X3) arrays for encoder predict
x_gd_product = np.zeros((4, iter))
x_gd_product[0, :] = x_gd[0, :] * x_gd[1, :]  # X1X2
x_gd_product[1, :] = x_gd[1, :] * x_gd[2, :]  # X2X3
x_gd_product[2, :] = x_gd[0, :] * x_gd[2, :]  # X1X3
x_gd_product[3, :] = x_gd[0, :] * x_gd[1, :] * x_gd[2, :]  # X1X2X3
# generate combined X gd (X1, X2, X3, X1**2, X2**2, X3**2, X1X2, X2X3, X1X3, X1X2X3) array
x_gd = np.concatenate((x_gd, x_gd ** 2, x_gd_product))
# transpose X gd
X_gd = np.transpose(x_gd)

##########################AUTOENCODER###################################
# ---------------------------Encoder------------------------------------#
model = Sequential()
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

Encoder = Model(input, hidden5)  # the output of encoder is batch normalization layer
model = Model(input, output)

# ---------------------------Decoder------------------------------------#
# assuming the model has 2 layers between the 2D layer and the output layer
# Decoder
encoded_input = Input(shape=(2,))
decoder1=model.layers[-4](encoded_input)
decoder2=model.layers[-3](decoder1)
decoder3=model.layers[-2](decoder2)
decoder4=model.layers[-1](decoder3)
Decoder=Model(encoded_input, decoder4)

# ----------------------Model Compile & Fit------------------------------#
# compile the model
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

# callback = [ EarlyStopping(monitor='val_loss', patience=2, min_delta=0.01 ,verbose=0) ]
# train the model with the train and validation data
# model.fit(X_train, y_train, epochs=epochs, verbose=2)

callback = [EarlyStopping(monitor='val_loss', patience=2, min_delta=0.01 ,verbose=0)]
model.fit(X_train, y_train, epochs=epochs, verbose=1, callbacks=callback)

##########################TEST DATASET####################################
# generate X test array
X_test = (b - a) * np.random.rand(3, N) + a
# set start values of each dimension to zero
X_test[0, 0] = 0
X_test[1, 0] = 0
X_test[2, 0] = 0
# combine X test and X optimal arrays
X_test = X_test + X_optimal
# generate X test products (X1X2, X2X3, X1X3, X1X2X3) arrays
X_testPro = np.zeros((4, N))
X_testPro[0, :] = X_test[0, :] * X_test[1, :]  # X1X2
X_testPro[1, :] = X_test[1, :] * X_test[2, :]  # X2X3
X_testPro[2, :] = X_test[0, :] * X_test[2, :]  # X1X3
X_testPro[3, :] = X_test[0, :] * X_test[1, :] * X_test[2, :]  # X1X2X3
# generate target function ||y-A*X||**2 for each dimension X1, X2, X3
f_test = (Y - np.dot(A, X_test)) ** 2
# generate target function ||y-A*X||**2
y_test = sum(f_test)
# sorting to find minimum corresponding index
index = np.argsort(y_test)
# generate combined X (X1, X2, X3, X1**2, X2**2, X3**2, X1X2, X2X3, X1X3, X1X2X3) array
x_test = np.concatenate((X_test, X_test ** 2, X_testPro))
# transpose X train
X_test = np.transpose(x_test)

##########################PREDICT ANSWERS####################################
# low dimensional output for X test
low_dim = Encoder.predict(X_test)
# find the first X with positive entries corresponding to the minimum value of fhat
i = 0
while low_dim[index[i], 0] <= 0 or low_dim[index[i], 1] <= 0:
    i += 1
# local minimum point
min_test_y = y_test[index[i]]
min_test_x1 = low_dim[index[i], 0]
min_test_x2 = low_dim[index[i], 1]
# low dimensional output for X gd (gradient descent)
low_dimGd = Encoder.predict(X_gd)

# ---------------------------2D Random Input----------------------------------#
X_fit = 2 * 6 * np.random.rand(2, N) - 6
# set starting points to zero
X_fit[0, 0] = 0
X_fit[1, 0] = 0
# transpose X fit
X_fit = np.transpose(X_fit)
# predict the target function for X fit
fhat = np.squeeze(Decoder.predict(X_fit))
# sorting to find minimum
# sorted_fhat=np.sort(fhat)
# sorting to find minimum corresponding index
index = np.argsort(fhat)
# find the first X with positive entries corresponding to the minimum value of fhat
i = 0
while X_fit[index[i], 0] <= 0 or X_fit[index[i], 1] <= 0:
    i += 1
# local minimum point
min_y = fhat[index[i]]
min_x1 = X_fit[index[i], 0]
min_x2 = X_fit[index[i], 1]

# triangularize the points
triang = tr.Triangulation(low_dim[:, 0], low_dim[:, 1])
triang1 = tr.Triangulation(X_fit[:, 0], X_fit[:, 1])

# calculate the accuracy using test data
score = model.evaluate(X_test, y_test, verbose=0)
# print the loss and accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#########################PLOT 2D & 3D FIGURES##############################
# -----------------------------2D Plot----------------------------------#
# plot the results
plt.figure(figsize=(9, 4))  # generate figure with a size of 9x4 inches
plt.subplot(121)  # subplot 1 row 2 columns the first item
plt.tricontourf(triang, np.squeeze(y_test))  # draw contour colors
plt.colorbar()  # draw colorbar
plt.tricontour(triang, np.squeeze(y_test))  # draw contour lines
plt.scatter(low_dimGd[:, 0], low_dimGd[:, 1], marker='o', c='r', s=10)  # red circle with size 10
plt.scatter(min_test_x1, min_test_x2, marker='^', c='g', s=20)
plt.title("True Objective")  # set title
plt.xlabel('Z1')
plt.ylabel('Z2')

fig = plt.subplot(122)  # subplot 1 row 2 columns the second item
plt.tricontourf(triang1, np.squeeze(fhat))  # draw contour colors
plt.colorbar()  # draw colorbar
plt.tricontour(triang1, np.squeeze(fhat))  # draw contour lines
plt.scatter(low_dimGd[:, 0], low_dimGd[:, 1], marker='o', c='r', s=10)  # red circle with size 10
plt.scatter(min_x1, min_x2, marker='^', c='g', s=20)
plt.title("NN Estimated Objective")
plt.xlabel('Z1')
plt.ylabel('Z2')

# ------------------------------3D Plot----------------------------------#
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_trisurf(low_dim[:, 0], low_dim[:, 1], np.squeeze(y_test), cmap=plt.cm.viridis, linewidth=0.2,
                       antialiased=True)  # cmap=cm.coolwarm
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.scatter(low_dimGd[:, 0], low_dimGd[:, 1], y_gd, c='r', marker='o')
ax.plot(low_dimGd[:, 0], low_dimGd[:, 1], y_gd, c='r')
ax.scatter(min_test_x1, min_test_x2, min_test_y, c='g', marker='h', s=20)
plt.title("True Objective")  # set title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_trisurf(X_fit[:, 0], X_fit[:, 1], np.squeeze(fhat), cmap=plt.cm.viridis, linewidth=0.2,
                antialiased=True)  # cmap=cm.coolwarm
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.scatter(low_dimGd[:, 0], low_dimGd[:, 1], y_gd, c='r', marker='o')
ax.plot(low_dimGd[:, 0], low_dimGd[:, 1], y_gd, c='r')
ax.scatter(min_x1, min_x2, min_y, c='g', marker='h', s=20)
plt.title("NN Estimated Objective")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()  # show plot
