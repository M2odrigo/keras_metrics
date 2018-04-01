# Create your first MLP in Keras
from desviation_function import calc_metric
from graph_metric_function import graph_metric
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np

#funcion para entrenar la red en N epoch y un batch size
def train_nn (X, Y, cant_epoch=[], batch=0):
    metric_zero=np.array([])
    metric_one = np.array([])
    for e in cant_epoch:
    #para poder reproducir los resultados, limitamos el "random" de las inicializaciones
        np.random.seed(7)
        e=int(e)
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='sigmoid'))
        model.add(Dense(8, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))

        #w_layer_one = model.layers[0].get_weights()
        #print('weights layer 0: ', w_layer_one)

        #print('weights layer 1')
        #print(model.layers[1].get_weights())

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(X, Y, epochs=e, batch_size=batch)
        # evaluate the model
        scores = model.evaluate(X, Y)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        print(' ')
        #print('####PREDICTION#####')
        #print(model.predict_proba(X))
    
        #extraer las activaciones
        ##hidden 1
        model2 = Sequential()
        model2.add(Dense(12, input_dim=8, weights=model.layers[0].get_weights(), activation='sigmoid'))
        activations2 = model2.predict_proba(X)
        a2 = np.asarray(activations2)
        desviation_one = calc_metric(1, 12, a2, Y)
        print("desviation_one",desviation_one)
        print("desviation_class0e",desviation_one[0])
        print("desviation_class1e",desviation_one[1])

        ##hidden 2
        model3 = Sequential()
        model3.add(Dense(8, input_dim=12, weights=model.layers[1].get_weights(), activation='sigmoid'))
        activations3 = model3.predict_proba(activations2)
        #print(activations3)
        a3 = np.asarray(activations3)
        desviation_two = calc_metric(2, 8, a3, Y)
        print("desviation_two",desviation_two)
        print("desviation_class0e",desviation_two[0])
        print("desviation_class1e",desviation_two[1])

        ##output
        model4 = Sequential()
        model4.add(Dense(1, input_dim=8, weights=model.layers[2].get_weights(), activation='sigmoid'))
        activations4 = model4.predict_proba(activations3)
        #print(activations4)
        a4 = np.asarray(activations4)
        desviation_three = calc_metric(3, 1, a4, Y)
        print("desviation_three",desviation_three)
        print("desviation_class0e",desviation_three[0])
        print("desviation_class1e",desviation_three[1])

        if(e == 0):
            initial_desviation_zero = np.concatenate((desviation_one[0], desviation_two[0], desviation_three[0]))
            initial_desviation_one = np.concatenate((desviation_one[1], desviation_two[1], desviation_three[1]))
            print("initial_desviation_zero", initial_desviation_zero)
            print("initial_desviation_one", initial_desviation_one)

        else:
            desviation_epoch_zero = np.concatenate((desviation_one[0], desviation_two[0], desviation_three[0]))
            desviation_epoch_one = np.concatenate((desviation_one[1], desviation_two[1], desviation_three[1]))
            print("desviation_epoch_zero", desviation_epoch_zero)
            print("desviation_epoch_one", desviation_epoch_one)
            print("metricas para epoch ---->", e)
            result_zero = np.divide(desviation_epoch_zero,initial_desviation_zero)
            result_one = np.divide(desviation_epoch_one,initial_desviation_one)
            print(result_zero)
            print(result_one)
            ##ir agregando las metricas por cada epoch
            if np.any(metric_zero):
                metric_zero= np.append(metric_zero, [result_zero], axis=0)
            else:
                metric_zero = [result_zero]
            print("metricas zero ", metric_zero)

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


#param_layers = input('cantidad capas ')
param_layers = 3
param_layers = np.arange(1, (int(param_layers)+1))
#param_epoch = input('cantidad epochs ')
param_epoch = "0,10,20"
param_epoch = param_epoch.split(',')

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

#normalizar los valores del train set
X = scaler.fit_transform(X)

initial_desviation=[]
train_nn(X, Y, param_epoch, batch_size)

