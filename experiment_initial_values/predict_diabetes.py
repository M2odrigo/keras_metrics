# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import csv
from keras import optimizers
import keras

loss = 'binary_crossentropy'
activation1 = 'sigmoid'
activation2 = 'sigmoid'
optimizer= 'adam'
kernel_initializer = 'uniform'

# fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

dataset = numpy.loadtxt("pima-indians-diabetes-test.csv", delimiter=",")
# split into input (X) and output (Y) variables
X_test = dataset[:,0:8]
Y_test= dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer= kernel_initializer, activation=activation1))
model.add(Dense(8, kernel_initializer=kernel_initializer, activation=activation1))
model.add(Dense(1, kernel_initializer=kernel_initializer, activation=activation2))

# Compile model
model.compile(loss= loss, optimizer=optimizer, metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=0, batch_size=10,  verbose=2)
# calculate predictions
print('####PREDICTION#####')

config = 'loss: '+loss + ' kernel_initializer: '+ str(kernel_initializer) +' activation1: ' + activation1 + ' activation2: ' + activation2 + ' optimizer: '+optimizer 

predictions = model.predict(X_test)
for index,p in enumerate(predictions):
    fields=[str(p),str(Y_test[index]), config]
    with open('prediction.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
