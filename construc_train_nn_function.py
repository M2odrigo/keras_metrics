
#funcion para construir la red y entrenarla
##X conjunto de entrada
##Y salida esperada
##cant_input cantidad de entradas que recibira el primer hidden, serian las neuronas del input layer
##cant_layers cantidad de capas desde el hidden1 hasta el output
##cant_epoch array con los epochs que realizara
##batch es opcional, establece de a cuantos lotes se leera el conjunto de entrada X
from desviation_function import calc_metric
from split_function import split
from sgd_perceptron import perceptron_plot
from graph_metric_function import graph_metric
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
import os

def construct_train_nn (X, Y, cant_input, cant_layers, cant_nodos, cant_epoch, batch=0):
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
    for e in cant_epoch:
        #np.random.seed(14)
        e=int(e)
        model = Sequential()
        for layer in indice_capas:
            if layer==0:
                model.add(Dense(int(cant_nodos[layer]), input_dim=int(cant_input),activation='sigmoid'))
            else:
                model.add(Dense(int(cant_nodos[layer]), activation='sigmoid'))

        print("Configuracion de la red: ", model.summary())

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("entrenando para ", e, " epochs")
        model.fit(X, Y, epochs=e, batch_size=int(batch))

        print('####PREDICTION#####')
        prediction = model.predict_proba(X_test)
        #for index,p in enumerate(prediction):
        #    fields=[str(e),str(p),str(Y_test[index])]
        #    with open('prediction.csv', 'a') as f:
        #        writer = csv.writer(f)
        #        writer.writerow(fields)

        # evaluate the model
        scores = model.evaluate(X, Y)
        acc = ("%.2f%%" % (scores[1]*100))
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
                    desviation = calc_metric(layer, int(cant_nodos[layer]), activations, Y)
                    initial_desviation_zero = desviation[0]
                    initial_desviation_one = desviation[1]
                else:
                    print("cant_nodos anterior: ", cant_nodos[layer-1])
                        
                    activations_layers = get_activations(int(cant_nodos[layer]), int(cant_nodos[layer-1]), model.layers[layer].get_weights(), activations)
                    activations = activations_layers
                    save_activation (e, cant_nodos[layer], activations, Y, layer, indice_capas[-1])
                    desviation = calc_metric(layer, int(cant_nodos[layer]), activations_layers, Y)
                    initial_desviation_zero = np.concatenate((initial_desviation_zero, desviation[0]))
                    initial_desviation_one = np.concatenate((initial_desviation_one, desviation[1]))
                    
                print("initial_desviation_zero", initial_desviation_zero)
                print("initial_desviation_one", initial_desviation_one)

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
                    desviation = calc_metric(layer, int(cant_nodos[layer]), activations, Y)
                    desviation_epoch_zero = desviation[0]
                    desviation_epoch_one = desviation[1]
                else:
                    print("cant_nodos anterior: ", cant_nodos[layer-1])
                        
                    activations_layers = get_activations(int(cant_nodos[layer]), int(cant_nodos[layer-1]), model.layers[layer].get_weights(), activations)
                    activations = activations_layers
                    save_activation (e, cant_nodos[layer], activations, Y, layer, indice_capas[-1])
                    desviation = calc_metric(layer, int(cant_nodos[layer]), activations_layers, Y)
                    desviation_epoch_zero = np.concatenate((desviation_epoch_zero, desviation[0]))
                    desviation_epoch_one = np.concatenate((desviation_epoch_one, desviation[1]))

            print("desviation_epoch_zero", desviation_epoch_zero)
            print("desviation_epoch_one", desviation_epoch_one)


            print("metricas para epoch ---->", e)
            result_zero = np.divide(desviation_epoch_zero,initial_desviation_zero)
            result_one = np.divide(desviation_epoch_one,initial_desviation_one)
            print(result_zero)
            print(result_one)
            ##ir agregando las metricas por cada epoch para la clase 0
            if np.any(metric_zero):
                metric_zero= np.append(metric_zero, [result_zero], axis=0)
            else:
                metric_zero = [result_zero]
            print("metricas zero ", metric_zero)
            ##ir agregando las metricas por cada epoch para la clase 1
            if np.any(metric_one):
                metric_one= np.append(metric_one, [result_one], axis=0)
            else:
                metric_one = [result_one]
            print("metricas one ", metric_one)

    if np.any(metric_zero):
        print ('vamos a graficar')
        print ([metric_zero,metric_one])
        #existe metrica one --> hay otra clase, duplicar el parametro "layers" asi su metrica se puede graph
        if np.any(metric_one):
            graph_metric([param_layers, param_layers], [metric_zero,metric_one], param_epoch)
        else:
            graph_metric(param_layers, metric_zero, param_epoch)

def get_activations (cant_nodos, cant_input, weights, activations):
    model = Sequential()
    model.add(Dense(cant_nodos, input_dim=cant_input, weights=weights, activation='sigmoid'))
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
        
        split(open('hidden_'+str(layer) +'_activations.csv', 'r'));
        filename1 = 'output_1.csv'
        filename2 = 'output_2.csv'
        perceptron_plot(filename1, layer, e,cant_nodos)
        perceptron_plot(filename2, layer, e,cant_nodos)
        #input('continue?') 
    else:
        print("capa actual " + str(layer))
        print("capa final " + str(last_layer))
        print("llegamos al output layer")

nro_capa = 1  
param_layers = 4
param_layers = np.arange(1, (int(param_layers)+1))
#param_epoch = input('cantidad epochs ')
param_epoch= "0,200"
#for i in np.arange(0,201,100):
#    if(param_epoch==""):
#        param_epoch = str(i)
#    else:
#        param_epoch = param_epoch + "," + str(i)
print(param_epoch)
param_epoch = param_epoch.split(',')
#cantidad de nodos del input layer
param_input = 8
#cantidad de nodos por cada capa, array de nodos
param_nodos = "12,8,8,1"
#param_nodos = "12"
param_nodos = param_nodos.split(',')
##variables
batch_size = 100
###
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
#normalize the data
scaler = StandardScaler()
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# load TEST pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes-test.csv", delimiter=",")
X_test = dataset[:,0:8]
Y_test = dataset[:,8]
#normalizar los valores del train set
#X = scaler.fit_transform(X)
construct_train_nn(X, Y, param_input, param_layers, param_nodos, param_epoch, batch_size)

