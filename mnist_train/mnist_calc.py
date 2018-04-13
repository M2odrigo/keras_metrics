# imports for array-handling and plotting
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# let's keep our keras backend tensorflow quiet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import csv
# for testing on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

from mnist_varianza import calc_varianza
from mnist_varianza import calc_media
from mnist_varianza import get_values_classes

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# let's print the shape before we reshape and normalize
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)

print(np.unique(y_train, return_counts=True))

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))   
#dropout helps avoid the overfittin by zero-ing some random input in each iteration                         
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
sum_class_zero = np.array([])
contador_zero = 0
param_epoch= "0"
param_epoch = param_epoch.split(',')
for e in param_epoch: 
    # training the model and saving metrics in history
    history = model.fit(X_train, Y_train,
              batch_size=128, epochs=int(e),
              verbose=2,
              validation_data=(X_test, Y_test))


    cont = 0
    print('####PREDICTION#####')
    prediction = model.predict_proba(X_test)
    for index,p in enumerate(prediction):
        cont = cont + 1
        if cont > 50:
            break
        r2 = [format(x, 'f') for x in p]
        print("r init " , r2)
        fields=[e,r2,str(Y_test[index]), str(Y_test[index].argmax())]
        with open('prediction.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    
    calculo_tmp = calc_media(prediction, '0', Y_test)
    media = calculo_tmp [0]
    contador = calculo_tmp[1]
    print("media_function", media, ' cont', contador)

    r = get_values_classes(prediction, '0', Y_test)
    print('r ', r)

    varianza = calc_varianza(r, media, 0, contador)
    print('varianza', varianza)
 

#print("#########outputs of input layer with 512 nodes#################")
#model2 = Sequential()
#model2.add(Dense(512, input_shape=(784,), weights=model.layers[0].get_weights(), activation='relu'))
#model2.add(Dropout(0.2))
#activations = model2.predict(X_test)
#print(activations)

#a = np.asarray(activations)

#roundnumpy = np.round(activations, 5)

#arra = np.asarray(activations)
#a = np.array_str(arra, precision=4)

#np.savetxt("512nodes_batch_100_epoch_5_round.csv", a, fmt = '%.5f', delimiter=",")

