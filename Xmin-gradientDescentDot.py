'''Keras Neural Networks Objective Function Regression f(x)=||y-Ax||^2
'''

# import necessary packages
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

epochs = 30
N = 12000  # number of points in the train set
N_test = 3000  # number of points in the test set
A = [[4,9,2], [8,1,6], [3,5,7]]

# Gradient Descent function realized
def gradientDescent(m,A,y,x_start,encoder,model,alpha):
    gd = np.empty((m, 3))
    for i in range(m):
        x_input = np.expand_dims(np.squeeze(np.transpose(x_start)), axis=0)
        z_pos = encoder.predict(x_input)
        y_output = model.predict(x_input)
        gd[i, :] = (np.concatenate((z_pos, y_output), axis=1))
        Axty = np.dot(A, x_start) - y
        x_start = x_start - alpha * 2 * np.dot(np.transpose(A), Axty)
    return gd

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

################### Decoder #################################################################################
encoded_input = Input(shape=(2,))
decoder1=model.layers[-4](encoded_input)
decoder2=model.layers[-3](decoder1)
decoder3=model.layers[-2](decoder2)
decoder4=model.layers[-1](decoder3)
Decoder=Model(encoded_input, decoder4)

################### Create testing data similarly ###########################################################
np.random.seed(12345)
X1test = 0.16 * np.random.rand(1, N_test) - 0.08
X2test = 1.6 * np.random.rand(1, N_test) - 0.8
X3test = 0.16 * np.random.rand(1, N_test) - 0.08

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

fnntest = np.transpose((y1nntest - z1nntest) ** 2 + (y2nntest - z2nntest) ** 2 + (y3nntest - z3nntest) ** 2)
Xtest = np.squeeze(np.transpose(Xtest))
low_dim=Encoder.predict(Xtest)# predict the objective function for Xtest, low dimensional output
triang=tr.Triangulation(low_dim[:,0],low_dim[:,1])

################### generate input for decoder ###################################################################
np.random.seed(12345)
X1dtest = 10000 * np.random.rand(1, N_test) - 5000
X2dtest = 10000 * np.random.rand(1, N_test) - 5000

Xdtest = np.concatenate((X1dtest, X2dtest))
Xdtest = np.squeeze(np.transpose(Xdtest))
triangd=tr.Triangulation(Xdtest[:,0],Xdtest[:,1])
dlow_dim = Decoder.predict(Xdtest)# one dimensional decoder output

################### gradient points ##################################################################################

A= np.array([[4, 9, 2], [8, 1, 6], [3, 5, 7]])
y=np.asarray([[8],[2],[5]])
pre_X=[[5],[5],[10]]
gd = gradientDescent(20,A,y,pre_X,Encoder,model,0.005)


#gd_out = Decoder.predict(trace)

################### mark the min ##################################################################################
# generate optimal points to mark the optimal point
X1marker = np.full((1, N),0.0833)
X2marker = np.full((1, N),0.833)
X3marker = np.full((1, N),0.0833)
# concetenate values into a matrix
Xmarker = np.concatenate((X1marker, X2marker, X3marker))
# modify X so it can be trainable
X_marker = np.squeeze(np.transpose(Xmarker))
Xmarker = np.squeeze(np.transpose(Xmarker))
Xmarker_ldim=Encoder.predict(Xmarker)# low dimensional output

################### plot the result ###########################################################################
# plt.figure(figsize=(9,4)) #generate figure with a size of 9x4 inches
# plt.subplot(121)#subplot 1 row 2 columns the first item
# plt.tricontourf(triang,np.squeeze(fnntest))#draw contour colors
# plt.colorbar()#draw colorbar
# plt.tricontour(triang,np.squeeze(fnntest))#draw contour lines
# plt.title("True Objective")#set title
#
# ax=plt.subplot(122)#subplot 1 row 2 columns the second item
# plt.tricontourf(triangd,dlow_dim)#draw contour colors
# plt.colorbar()#draw colorbar
# plt.tricontour(triang,np.squeeze(dlow_dim))#draw contour lines
# plt.title("NN Estimated Objective")
#
# ax.annotate(np.squeeze(trace[:,0]), np.squeeze(trace[:,1]), arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
#                      va='center', ha='center')
#
# plt.show()#show plot (this blocks the code: make sure it's at the end of your code)


fig = plt.figure()
ax = fig.gca(projection='3d')
surf1 = ax.plot_trisurf(triang, np.squeeze(fnntest), cmap=plt.cm.viridis,
                      linewidth=0, antialiased=False)
fig.colorbar(surf1, shrink=0.5, aspect=5) # Add a color bar which maps values to colors.
plt.title("True Objective")  # set title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(np.squeeze(np.asarray(gd[:,0])), np.squeeze(np.asarray(gd[:,1])),np.squeeze(np.asarray(gd[:,2])), c='r', marker='o')

#ax.annotate(np.squeeze(trace[:,0]), np.squeeze(trace[:,1]),np.squeeze(trace[:,2]), arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
#                    va='center', ha='center')
surf2 = ax.plot_trisurf(triangd, np.squeeze(dlow_dim), cmap=plt.cm.viridis,
                       linewidth=0, antialiased=False)
fig.colorbar(surf2, shrink=0.5, aspect=5)# Add a color bar which maps values to colors.
plt.title("NN Estimated Objective")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()  # show plot (this blocks the code: make sure it's at the end of your code)


