
#funcion para construir la red y entrenarla
##X conjunto de entrada
##Y salida esperada
##cant_input cantidad de entradas que recibira el primer hidden, serian las neuronas del input layer
##cant_layers cantidad de capas desde el hidden1 hasta el output
##cant_epoch array con los epochs que realizara
##batch es opcional, establece de a cuantos lotes se leera el conjunto de entrada X

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import os
import pandas

def construct_train_nn (X, Y, cant_input, cant_layers, cant_nodos, cant_epoch, batch=0, X_test = '', y_test =''):
    if os.path.isfile('hidden_perceptron_error.csv'):
        os.remove('hidden_perceptron_error.csv')
    indice_capas = np.arange((np.count_nonzero(cant_layers)))
    print("recibimos capas ", indice_capas)
    metric_zero=np.array([])
    metric_one = np.array([])
    initial_desviation_zero=np.array([])
    initial_desviation_one = np.array([])
    desviation_epoch_zero = np.array([])
    desviation_epoch_one = np.array([])
    activations = []
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=seed)
    for e in cant_epoch:
        #np.random.seed(14)
        e=int(e)
        model = Sequential()
        for layer in indice_capas:
            if layer==0:
                model.add(Dense(int(cant_nodos[layer]), input_dim=int(cant_input),activation='relu'))
            else:
                if(layer != indice_capas[-1]):
                    model.add(Dense(int(cant_nodos[layer]), activation='relu'))
                else:
                    model.add(Dense(int(cant_nodos[layer]), activation='sigmoid'))

        print("Configuracion de la red: ", model.summary())
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("entrenando para ", e, " epochs")
        model.fit(X, Y, epochs=e, batch_size=int(batch))

        print('####PREDICTION#####')
        prediction = model.predict_proba(X)

        # evaluate the model
        scores = model.evaluate(X_test, y_test)
        print(scores)
        acc = ("%.2f%%" % (scores[1]*100))  
        print('acc::: ', acc)
        row = str(e)+","+str(acc)
        fields=[str(e),str(acc)]
        # 'a' para agregar contenido al CSV
        # 'wb' para sobre-escribir el archivo CSV
        with open('epoch_acc.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        if(e == 0):
            #extraer las activaciones
            for layer in indice_capas:
                print("###extraer las activaciones y calcular la desviacion por clase")
                print("layer: ", layer)
                print("cant_nodos: ", cant_nodos[layer])
                print("cant input: ", cant_input)
                if layer==0:
                    activations = get_activations(int(cant_nodos[layer]), cant_input, model.layers[layer].get_weights(), X)
                    print('###############')
                    print(activations)
                    print("activation> ", activations.shape)
                    print("Y> ", Y.shape)
                    save_activation (e, cant_nodos[layer], activations, Y, layer, indice_capas[-1])

                else:
                    print("cant_nodos anterior: ", cant_nodos[layer-1])
                        
                    activations_layers = get_activations(int(cant_nodos[layer]), int(cant_nodos[layer-1]), model.layers[layer].get_weights(), activations)
                    activations = activations_layers
                    save_activation (e, cant_nodos[layer], activations, Y, layer, indice_capas[-1])
                    desviation = calc_metric(layer, int(cant_nodos[layer]), activations_layers, Y)

        else:
            #extraer las activaciones
            for layer in indice_capas:
                print("###extraer las activaciones y calcular la desviacion por clase")
                print("layer: ", layer)
                print("cant_nodos: ", cant_nodos[layer])
                print("cant input: ", cant_input)
                if layer==0:
                    activations = get_activations(int(cant_nodos[layer]), cant_input, model.layers[layer].get_weights(), X)
                    save_activation (e, cant_nodos[layer], activations, Y, layer, indice_capas[-1])
                else:
                    print("cant_nodos anterior: ", cant_nodos[layer-1])
                        
                    activations_layers = get_activations(int(cant_nodos[layer]), int(cant_nodos[layer-1]), model.layers[layer].get_weights(), activations)
                    activations = activations_layers
                    save_activation (e, cant_nodos[layer], activations, Y, layer, indice_capas[-1])

def get_activations (cant_nodos, cant_input, weights, activations):
    model = Sequential()
    model.add(Dense(cant_nodos, input_dim=cant_input, weights=weights, activation='relu'))
    activations = model.predict_proba(activations)
    activations_array = np.asarray(activations)
    return activations_array

def save_activation (e, cant_nodos, activations, Y, layer, last_layer):
    if(layer != last_layer):
        print("epoch ", e, "cant nodos: ", cant_nodos, "activation shape: ", activations.shape)
        #print(activations)
        if os.path.isfile('hidden_'+str(layer) +'_activations.csv'):
            os.remove('hidden_'+str(layer) +'_activations.csv')
        for index, activ in enumerate(Y):
            r2 = ['{:f}'.format(x) for x in activations[index]]
            fields=[r2, Y[index]]
            with open('hidden_'+str(layer) +'_activations.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)

        # Read in the file
        with open('hidden_'+str(layer) +'_activations.csv', 'r') as file :
            filedata = file.read()
            # Replace the target string
            filedata = filedata.replace('[', '')
            filedata = filedata.replace(']', '')
            filedata = filedata.replace('"', '')
            filedata = filedata.replace('\'', '')
        # Write the file out again
        with open('hidden_'+str(layer) +'_activations.csv', 'w') as file:
            file.write(filedata)
        
        #split(open('hidden_'+str(layer) +'_activations.csv', 'r'));
        filename1 = 'output_1.csv'
        filename2 = 'output_2.csv'
        #perceptron_plot(filename1, layer, e,cant_nodos)
        #perceptron_plot(filename2, layer, e,cant_nodos)
        #input('continue?') 
    else:
        print("capa actual " + str(layer))
        print("capa final " + str(last_layer))
        print("llegamos al output layer")


param_layers = 2
param_layers = np.arange(1, (int(param_layers)+1))

param_epoch= "100"
print(param_epoch)
param_epoch = param_epoch.split(',')

#cantidad de nodos del input layer
param_input = 60
#cantidad de nodos por cada capa, array de nodos
param_nodos = "30,1"
param_nodos = param_nodos.split(',')

##variables
batch_size = 100
###

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("sonar-train.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

# encode class values as integers, convierte el output actual (R/M) en clases binarias (0/1)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# load test dataset
dataframes = pandas.read_csv("sonar-test.csv", header=None)
datasets = dataframes.values
# split into input (X) and output (Y) variables
X_test = datasets[:,0:60].astype(float)
Y_test = datasets[:,60]

encodert = LabelEncoder()
encodert.fit(Y_test)
encoded_Y_test = encodert.transform(Y_test)

#normalize the data
#scaler = StandardScaler()
#normalizar los valores del train set
#X = scaler.fit_transform(X)
#X_test =  scaler.fit_transform(X_test)

construct_train_nn(X, encoded_Y, param_input, param_layers, param_nodos, param_epoch, batch_size, X_test, encoded_Y_test)

