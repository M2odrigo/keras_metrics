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

from graph_metric_function import graph_metric


def get_activations (cant_nodos, cant_input, weights, activations):
    model = Sequential()
    model.add(Dense(cant_nodos, input_dim=cant_input, weights=weights, activation='sigmoid'))
    activations = model.predict_proba(activations)
    activations_array = np.asarray(activations)
    return activations_array


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
model.add(Dense(512, input_shape=(784,), activation='relu'))
#model.add(Activation('relu'))   
#dropout helps avoid the overfittin by zero-ing some random input in each iteration                         
#model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
contador_zero = 0
param_epoch= "0,1,2,3"
param_epoch = param_epoch.split(',')
param_calses = "0,1,2,3,4,5,6,7,8,9"
param_calses = param_calses.split(',')
varianza_epoch_zero = np.array([])
varianza_classes = np.array([])
list_varianzas = []
list_metrics = []
for e in param_epoch: 
    # training the model and saving metrics in history
    history = model.fit(X_train, Y_train,
              batch_size=128, epochs=int(e),
              verbose=2,
              validation_data=(X_test, Y_test))

    activations1 = get_activations(512, 784, model.layers[0].get_weights(), X_train)
    activations2 = get_activations(512, 512, model.layers[1].get_weights(), activations1)
    print('activations2',activations2.shape)
    print('y train ', Y_train.shape)

    cont = 0
    print('####PREDICTION#####')
    prediction = model.predict_proba(X_test)
    print('prediction ',prediction.shape) 
    for index,p in enumerate(prediction):
        cont = cont + 1
        if cont > 5000:
            break
        r2 = [format(x, 'f') for x in p]
        #print("r init " , r2)
        fields=[e,r2,str(Y_test[index]), str(Y_test[index].argmax())]
        with open('prediction.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    for cl in param_calses:
        cl=str(cl)
        print("vamos a calcular para la clase ", cl)
        calculo_tmp = calc_media(activations2, cl, Y_train)
        media = calculo_tmp [0]
        contador = calculo_tmp[1]
        print("media_function", media, ' cont', contador)

        r = get_values_classes(activations2, cl, Y_train)
        #print('r ', r)

        varianza = calc_varianza(r, media, cl, contador)
        print('varianza class ',cl , "equals to: ", varianza)
        print("epoch : ", e)
        if(str(e) == '0'):
            if np.any(varianza_epoch_zero):
                varianza_epoch_zero= np.append(varianza_epoch_zero, [varianza], axis=0)
            else:
                varianza_epoch_zero = [varianza]
        else:
            list_varianzas.append([varianza])
            #if np.any(varianza_classes):
            #    varianza_classes= np.append(varianza_classes, [varianza], axis=0)
            #else:
             #   varianza_classes = np.array([varianza])

        varianza_classes = np.asarray(list_varianzas)
        print ('varianza_epoch_zero >>>> ', varianza_epoch_zero)
        print ('varianza_classes >>>> ', varianza_classes)

    print("END")
    print ('varianza_epoch_zero >>>> ', varianza_epoch_zero)
    print ('varianza_classes >>>> ', varianza_classes)
tmpv = np.reshape(varianza_classes, (-1, 10))
print ('varianza_classes >>>> ', tmpv)
varianza_classes = tmpv
for index, v in enumerate(varianza_classes): 
    print('v : ', varianza_classes[index])
    print('varianza_epoch_zero : ', varianza_epoch_zero)
    metric = np.divide(varianza_classes[index], varianza_epoch_zero)
    list_metrics.append([metric])
    r1 = [format(x, 'f') for x in metric]
    r1 = [float(x) for x in metric]
    fields=[(index+1), r1]
    with open('metrcis.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

metrics = np.divide(varianza_classes, varianza_epoch_zero)
list_metrics = np.reshape(list_metrics, (-1, 10))
print('list metrics ',list_metrics)
param_layers = np.arange(1, (int(3)+1))
print ('param_layers ', param_layers)


#graph_metric([param_layers, param_layers,param_layers,param_layers,param_layers,param_layers,param_layers,param_layers,param_layers,param_layers], list_metrics, param_epoch)


#arra = np.asarray(list_metrics)
#a = np.array_str(arra, precision=6)
#np.savetxt("metrcs_.csv", a, fmt = '%.6f', delimiter=",")

