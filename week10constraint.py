
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tr
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow warnings

np.random.seed(21345)
epochs = 30
N = 12000  # number of points in the train set
N_test = 3000  # number of points in the test set
A = [[4, 9, 2], [8, 1, 6], [3, 5, 7]]

#################################Gradient Descent function#############################################################
def gradientDescent(m, A, y, x_start, encoder, model, alpha):
    gd = np.empty((m, 3))
    for i in range(m):
        x_input = np.expand_dims(np.squeeze(np.transpose(x_start)), axis=0)
        z_pos = encoder.predict(x_input)
        y_output = model.predict(x_input)
        gd[i, :] = (np.concatenate((z_pos, y_output), axis=1))
        Axty = np.dot(A, x_start) - y
        x_start = x_start - alpha * 2 * np.dot(np.transpose(A), Axty)

        for j in x_start :
            i = max(0, i)

    return gd

################################Generate points #######################################################################
X1 = 0.1666 * np.random.rand(1, N) - 0.0833
X2 = 1.666 * np.random.rand(1, N) - 0.833
X3 = 0.1666 * np.random.rand(1, N) - 0.0833

# concetenate values into a matrix
X = np.concatenate((X1, X2, X3))

# get the coordinates for each point
x1 = np.asarray(X[0, :])
x2 = np.asarray(X[1, :])
x3 = np.asarray(X[2, :])

# set Z=A*x for x1 and x2
Z = np.dot(A, X)
z1nn = np.asarray(Z[0, :])
z2nn = np.asarray(Z[1, :])
z3nn = np.asarray(Z[2, :])

# set y
y1nn = np.full((1, N), 8)
y2nn = np.full((1, N), 2)
y3nn = np.full((1, N), 5)

# calculate the function f(x)=||y-Ax||^2
fnn = np.transpose((y1nn - z1nn) ** 2 + (y2nn - z2nn) ** 2 + (y3nn - z3nn) ** 2)

# constraint, x returns a big value if x_i<0 and returns 0 elsewhere
for i in range(len(X)):
    for j in range(len(X[i])):
        if X[i][j] < 0:
            fnn[i] = 100000

# modify X so it can be trainable
X_train = np.transpose(X)

################################ Create the model ######################################################################
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
output = Dense(1, activation='linear')(hidden8)

# Encoder
Encoder = Model(input, hidden5)
model = Model(input, output)

# Decoder
encoded_input = Input(shape=(2,))
decoder1 = model.layers[-4](encoded_input)
decoder2 = model.layers[-3](decoder1)
decoder3 = model.layers[-2](decoder2)
decoder4 = model.layers[-1](decoder3)
Decoder = Model(encoded_input, decoder4)

# compile the model
model.compile(loss='mean_squared_error', optimizer='rmsprop')

# Early stopping and fit data
callback = [EarlyStopping(monitor='val_loss', patience=2, min_delta=0.01, verbose=0)]
model.fit(X_train, fnn, epochs=epochs, verbose=1, validation_split=0.25, callbacks=callback)

################### Create testing data #####################################################################
X1test = 0.1666 * np.random.rand(1, N_test) - 0.0833
X2test = 1.666 * np.random.rand(1, N_test) - 0.833
X3test = 0.1666 * np.random.rand(1, N_test) - 0.0833

Xtest = np.concatenate((X1test, X2test, X3test))
x1test = np.asarray(Xtest[0, :])
x2test = np.asarray(Xtest[1, :])
x3test = np.asarray(Xtest[2, :])

Ztest = np.dot(A, Xtest)
z1nntest = np.asarray(Ztest[0, :])
z2nntest = np.asarray(Ztest[1, :])
z3nntest = np.asarray(Ztest[2, :])

y1nntest = np.full((1, N_test), 8)
y2nntest = np.full((1, N_test), 2)
y3nntest = np.full((1, N_test), 5)

fnntest = np.transpose((y1nntest - z1nntest) ** 2 + (y2nntest - z2nntest) ** 2 + (y3nntest - z3nntest) ** 2)

# constraint, x returns a big value if x_i<0 and returns 0 elsewhere
for i in range(len(Xtest)):
    for j in range(len(Xtest[i])):
        if Xtest[i][j] < 0:
            fnntest[i] = 100000
            break

Xtest = np.squeeze(np.transpose(Xtest))

################### predict #######################################################################################
# predict the objective function for Xtest
low_dim = Encoder.predict(Xtest)

# gradient predict from decoder
X1dtest = 100 * np.random.rand(1, N_test) - 25
X2dtest = 100 * np.random.rand(1, N_test) - 25

Xdtest = np.concatenate((X1dtest, X2dtest))
Xdtest = np.squeeze(np.transpose(Xdtest))
triangd = tr.Triangulation(Xdtest[:, 0], Xdtest[:, 1])
dlow_dim = Decoder.predict(Xdtest)  # one dimensional decoder output

triang = tr.Triangulation(low_dim[:, 0], low_dim[:, 1])

################### gradient##########################################################################################
A = np.array([[4, 9, 2], [8, 1, 6], [3, 5, 7]])
y = np.asarray([[8], [2], [5]])
pre_X = [[55], [10], [23]]
lr = 0.0001
iter = 50
gd = gradientDescent(iter, A, y, pre_X, Encoder, model, lr)

# returns z if z>0 and returns 0 otherwise
for i in range(len(gd)):
    for j in range(len(gd[i])):
        if gd[i][j] < 0:
            gd[i][j] = 0

print(gd)

################### plot the result ###########################################################################
# plt.figure(figsize=(4,4)) #generate figure with a size of 9x4 inches
# # ax=plt.subplot(121)#subplot 1 row 2 columns the first item
# # plt.tricontour(triang, np.squeeze(fnntest))#draw contour lines
# # plt.colorbar()#draw colorbar
# # plt.tricontour(triang, np.squeeze(fnntest))#draw contour lines
# # plt.plot(gd[:,0], gd[:,1], 'r')
# # plt.title("True Objective")#set title
#
# ax=plt.subplot(122)#subplot 1 row 2 columns the second item
# plt.tricontour(triangd, np.squeeze(dlow_dim))#draw contour lines
# plt.colorbar()#draw colorbar
# plt.tricontour(triangd, np.squeeze(dlow_dim))#draw contour lines
# plt.plot(gd[:,0], gd[:,1], 'r')
# plt.title("NN Estimated Objective")
#
# plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(np.squeeze(np.asarray(gd[:, 0])), np.squeeze(np.asarray(gd[:, 1])), np.squeeze(np.asarray(gd[:, 2])), c='r',
           marker='o')
ax.plot(np.squeeze(np.asarray(gd[:, 0])), np.squeeze(np.asarray(gd[:, 1])), np.squeeze(np.asarray(gd[:, 2])), c='r')
surf1 = ax.plot_trisurf(triang, np.squeeze(fnntest), cmap=plt.cm.viridis,
                        linewidth=0, antialiased=False)
fig.colorbar(surf1, shrink=0.5, aspect=5)  # Add a color bar which maps values to colors.
plt.title("True Objective")  # set title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(np.squeeze(np.asarray(gd[:, 0])), np.squeeze(np.asarray(gd[:, 1])), np.squeeze(np.asarray(gd[:, 2])), c='r',
           marker='o')
ax.plot(np.squeeze(np.asarray(gd[:, 0])), np.squeeze(np.asarray(gd[:, 1])), np.squeeze(np.asarray(gd[:, 2])), c='r')
surf2 = ax.plot_trisurf(triangd, np.squeeze(dlow_dim), cmap=plt.cm.viridis,
                        linewidth=0, antialiased=False)
fig.colorbar(surf2, shrink=0.5, aspect=5)  # Add a color bar which maps values to colors.
plt.title("NN Estimated Objective")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()  # show plot (this blocks the code: make sure it's at the end of your code)
