'''Keras Neural Networks Objective Function Regression f(x)=||y-Ax||^2
'''

# import necessary packages
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import SGD

from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from mpl_toolkits.mplot3d import axes3d, Axes3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tr
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow warnings

# number of iterations
epochs = 30

N = 10000  # number of points in the train set
N_test = 3000  # number of points in the test set

# generate train and validation points around minimum [0.0833， 0.833， 0.0833]
X1 = 0.0833 * np.random.rand(1, N)
X2 = 0.833 * np.random.rand(1, N)
X3 = 0.0833 * np.random.rand(1, N)

# generate optimal points for marker
X1marker = np.full((1, N),0.0833)
X2marker = np.full((1, N),0.833)
X3marker = np.full((1, N),0.0833)

# concetenate values into a matrix
X = np.concatenate((X1, X2, X3, X1*X1, X2*X2, X3*X3, X1*X2, X2*X3, X1*X3, X1*X2*X3))
XX = np.concatenate((X1, X2, X3))
Xmarker = np.concatenate((X1marker, X2marker, X3marker))

# get the coordinates for each point
x1 = np.asarray(X[0, :])
x2 = np.asarray(X[1, :])
x3 = np.asarray(X[2, :])

# set Z=A*x for x1 and x2
A = [[4,9,2], [8,1,6], [3,5,7]]

Z = np.dot(A, XX)

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
hidden1 = Dense(60, activation='relu')(input)
hidden2 = Dense(60, activation='relu')(hidden1)
hidden3 = Dense(60, activation='relu')(hidden2)
hidden4 = Dense(2, activation='relu')(hidden3)
hidden5 = BatchNormalization()(hidden4)

hidden6 = Dense(30, activation='relu')(hidden5)
hidden7 = Dense(30, activation='relu')(hidden6)
hidden8 = Dense(30, activation='relu')(hidden7)
output = Dense(1, activation='linear')(hidden8)

# Encoder
Encoder = Model(input, hidden5)
model = Model(input, output)

# compile the model
# learning_rate = 0.05
# decay_rate = learning_rate / epochs
# sgd = SGD(lr=learning_rate, momentum=0.8, decay=decay_rate, nesterov=False)
model.compile(loss='mean_absolute_error',optimizer='Rmsprop', metrics=['accuracy'])

# Fit the model, early stop on overfitting
callback = [EarlyStopping(monitor='val_loss', patience=2, min_delta=0.01 ,verbose=0)]
model.fit(X_train, fnn, epochs=epochs, verbose=2, validation_split=0.3, callbacks=callback)

# keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
#                             write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
#                             embeddings_metadata=None)

# generate the test data similarly
X1test = np.random.rand(1, N_test) * 0.0833
X2test = np.random.rand(1, N_test) * 0.833
X3test = np.random.rand(1, N_test) * 0.0833

XXtest = np.concatenate((X1test, X2test, X3test))
Xtest = np.concatenate((X1test, X2test, X3test, X1test*X1test, X2test*X2test, X3test*X3test, X1test*X2test, X1test*X3test,X2test*X3test,X1test*X2test*X3test))

x1test = np.asarray(Xtest[0, :])
x2test = np.asarray(Xtest[1, :])
x3test = np.asarray(Xtest[2, :])

Ztest = np.dot(A, XXtest)

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
fhat = np.squeeze(model.predict(Xtest))

Xmarker = np.squeeze(np.transpose(Xmarker))

# predict the objective function for X_test
low_dim=Encoder.predict(Xtest)# low dimensional output
#Xmarker_ldim=Encoder.predict(Xmarker)# low dimensional output
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_trisurf(low_dim[:,0],low_dim[:,1], np.squeeze(fnntest), linewidth=0.2, antialiased=True)
#
# plt.title("True Objective")  # set title
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_trisurf(low_dim[:,0],low_dim[:,1], fhat, linewidth=0.2, antialiased=True)
#
# plt.title("NN Estimated Objective")
# plt.show()  # show plot (this blocks the code: make sure it's at the end of your code)


#triangularize the points
triang=tr.Triangulation(x1test,x2test)

#plot the results
plt.figure(figsize=(9,4)) #generate figure with a size of 9x4 inches
plt.subplot(121)#subplot 1 row 2 columns the first item
plt.tricontourf(triang,np.squeeze(fnntest))#draw contour colors
plt.colorbar()#draw colorbar
plt.tricontour(triang,np.squeeze(fnntest))#draw contour lines
plt.title("True Objective")#set title
#plt.plot(Xmarker_ldim[0,0], Xmarker_ldim[0,1], '-rD')

plt.subplot(122)#subplot 1 row 2 columns the second item
plt.tricontourf(triang,fhat)#draw contour colors
plt.colorbar()#draw colorbar
plt.tricontour(triang,fhat)#draw contour lines
plt.title("NN Estimated Objective")
#plt.plot(Xmarker_ldim[0,0], Xmarker_ldim[0,1], '-rD')
plt.show()#show plot (this blocks the code: make sure it's at the end of your code)
