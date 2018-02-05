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


#seed of random data generation
np.random.seed(20171)#20172435 #2017243
#disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#number of iterations
epochs = 30
callback = [EarlyStopping(monitor='val_loss', patience=2, min_delta=0.01 ,verbose=0)]
A = [[1,0,0],
     [0,1,0],
     [0,0,1]]
b = [1,
     2,
     3]
N = 10000
Ntest = 2000
lim = 5
ul = 5
ll = -5

X=(ul - ll) * np.random.rand(3,N) + ll
Y=np.transpose(b * np.ones((N,3)))
f=(Y - np.dot(A,X))
ftrain = f[0]**2 + f[1]**2 + f[2]**2

for i in range(len(X)):
    for j in range(len(X[i])):
        #print(X[i][j])
        if X[i][j] < 0:
            ftrain[j] = 100000

X_train=np.transpose(X)

###################GD###########################
iter=50
#learning rate
tau=0.00001
#generate gradient descent starting points arrays
x_gd=np.zeros((3,iter))
x_gd[0,0]=1.1
x_gd[1,0]=2.1
x_gd[2,0]=3.1

for i in range(1,iter-1):
    x_gd[:, i] = x_gd[:, i - 1] - tau * 2 * np.dot(np.transpose(A), np.dot(A, x_gd[:, i - 1]) - b)
    if x_gd[0,i] < 0:
        x_gd[0, i] = -x_gd[0,i]
    if x_gd[1,i] < 0:
        x_gd[1, i] = -x_gd[1,i]
    if x_gd[2,i] < 0:
        x_gd[2, i] = -x_gd[2,i]


##########################AUTOENCODER###################################
#---------------------------Encoder------------------------------------#
# instantiate model
model = Sequential()

#the input layer
input=Input(shape=(X_train.shape[1],))

#the hidden layer
hidden1=Dense(100, activation='relu')(input)
hidden2=Dense(100, activation='relu')(hidden1)
hidden3=Dense(100, activation='relu')(hidden2)
hidden4=Dense(2,activation='relu')(hidden3)
bn=BatchNormalization()(hidden4)
hidden5=Dense(100,activation='relu')(bn)
hidden6=Dense(100,activation='relu')(hidden5)
hidden7=Dense(100,activation='relu')(hidden6)

#the output layer
output=Dense(1, activation='linear')(hidden7)

Encoder=Model(input,bn)#the output of encoder is batch normalization layer
model = Model(input,output)

#---------------------------Decoder------------------------------------#
# assuming the model has 2 layers between the 2D layer and the output layer
#Decoder
decoded_input = Input(shape=(2,))
decoder1=model.layers[-4](decoded_input)
decoder2=model.layers[-3](decoder1)
decoder3=model.layers[-2](decoder2)
decoder4=model.layers[-1](decoder3)

Decoder=Model(decoded_input, decoder4)

#----------------------Model Compile & Fit------------------------------#
#compile the model
model.compile(loss='mean_squared_error', optimizer= 'rmsprop')

#callback = [ EarlyStopping(monitor='val_loss', patience=2, min_delta=0.01 ,verbose=0) ]
#train the model with the train and validation data
model.fit(X_train, ftrain, epochs=epochs, verbose=2, validation_split=0.2, callbacks = callback)
#######################TEST################################

Xtest=(ul - ll) * np.random.rand(3,Ntest) + ll
Ytest=np.transpose(b * np.ones((Ntest,3)))
f=(Ytest - np.dot(A,Xtest))
ftest = f[0]**2 + f[1]**2 + f[2]**2

for i in range(len(Xtest)):
    for j in range(len(Xtest[i])):
        #print(X[i][j])
        if Xtest[i][j] < 0:
            ftest[j] = 100000

X_test=np.transpose(Xtest)

#############################Prdict############################
low_dim=Encoder.predict(X_test)
low_dimGd=Encoder.predict(np.transpose(x_gd))
Xes=(ul - ll) * np.random.rand(2,Ntest) + ll
Xes = np.transpose(Xes)
fhat=np.squeeze(Decoder.predict(Xes))

triang=tr.Triangulation(low_dim[:,0], low_dim[:,1])
triang1=tr.Triangulation(Xes[:,0], Xes[:,1])

################33333333Draw######################
plt.figure(figsize=(9,4)) #generate figure with a size of 9x4 inches
plt.subplot(121)#subplot 1 row 2 columns the first item
plt.tricontourf(triang,np.squeeze(ftest))#draw contour colors
plt.colorbar()#draw colorbar
plt.tricontour(triang,np.squeeze(ftest))#draw contour lines
plt.plot(low_dimGd[:,0], low_dimGd[:,1], 'r*-')  # red circle with size 10
plt.title("True Objective")#set title
plt.xlabel('Z1')
plt.ylabel('Z2')

fig = plt.subplot(122)#subplot 1 row 2 columns the second item
plt.tricontourf(triang1,np.squeeze(fhat))#draw contour colors
plt.colorbar()#draw colorbar
plt.tricontour(triang1,np.squeeze(fhat))#draw contour lines
plt.plot(low_dimGd[:,0], low_dimGd[:,1], 'r*-')  # red circle with size 10
plt.title("NN Estimated Objective")
plt.xlabel('Z1')
plt.ylabel('Z2')

plt.show()



