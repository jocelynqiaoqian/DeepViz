
'''Keras Neural Networks Objective Function Regression f(x)=||x||^2
2017 Emre Gonultas
'''

#import necessary packages
# note that all of these packages must be installed beforehand
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tr
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #disable tensorflow warnings

#number of iterations
epochs = 20

lim1=10#limit the range of values
N=10000 #number of points in the train set
N_test=1000# number of points in the test set

#randomly generate points between lim and -lim
a=-lim1
b=lim1
X1=(b-a)*np.random.rand(1,N)+a
X2=(b-a)*np.random.rand(1,N)+a

#concetenate values into a matrix
X=np.concatenate((X1,X2))

#set the corners equal to |10|
X[:,0]=[-lim1, -lim1]
X[:,1]=[-lim1, lim1]
X[:,2]=[lim1, -lim1]
X[:,3]=[lim1, lim1]

#get the coordinates for each point
x1=np.asarray(X[0,:])
x2=np.asarray(X[1,:])

#set A*x=z for x1 and x2
#A is the identity matrix in this case
z1nn=x1
z2nn=x2
#set y=0
y1nn= np.zeros((1, N))
y2nn= np.zeros((1, N))

#calculate the objective function for eac point
fnn=np.transpose((y1nn-z1nn)**2+(y2nn-z2nn)**2)

#modify X so it can be trainable
X_train=np.squeeze(np.transpose(X))

#create the model
model = Sequential()
#the first hidden layer
model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],))) #input shape is 1 in this case, because we only have 1 feature i.e. x
#the output layer with one neuron and a linear activation function
model.add(Dense(1, activation='linear'))

#compile the model
model.compile(loss='mean_squared_error',
              optimizer='rmsprop')
#train the model with the train and validation data
model.fit(X_train, fnn, epochs=epochs, verbose=2)


#generate the test data similarly
X1test=(b-a)*np.random.rand(1,N_test)+a
X2test=(b-a)*np.random.rand(1,N_test)+a

Xtest=np.concatenate((X1test,X2test))

Xtest[:,0]=[-lim1, -lim1]
Xtest[:,1]=[-lim1, lim1]
Xtest[:,2]=[lim1, -lim1]
Xtest[:,3]=[lim1, lim1]

x1test=np.asarray(Xtest[0,:])
x2test=np.asarray(Xtest[1,:])

z1nntest=x1test
z2nntest=x2test
y1nntest= np.zeros((1, N_test))
y2nntest= np.zeros((1, N_test))

#calculate objective function for the test data
fnntest=np.transpose((y1nntest-z1nntest)**2+(y2nntest-z2nntest)**2)

#modify X_test so it can be fed to the network
Xtest=np.squeeze(np.transpose(Xtest))
fhat=np.squeeze(model.predict(Xtest)) #predict the objective function for X_test

#triangularize the points
triang=tr.Triangulation(x1test,x2test)

#plot the results
plt.figure(figsize=(9,4)) #generate figure with a size of 9x4 inches
plt.subplot(121)#subplot 1 row 2 columns the first item
plt.tricontourf(triang,np.squeeze(fnntest))#draw contour colors
plt.colorbar()#draw colorbar
plt.tricontour(triang,np.squeeze(fnntest))#draw contour lines
plt.title("True Objective")#set title

plt.subplot(122)#subplot 1 row 2 columns the second item
plt.tricontourf(triang,fhat)#draw contour colors
plt.colorbar()#draw colorbar
plt.tricontour(triang,fhat)#draw contour lines
plt.title("NN Estimated Objective")
plt.show()#show plot (this blocks the code: make sure it's at the end of your code)
