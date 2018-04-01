import sys
param_epoch = sys.argv[1]

train_nn(param_epoch)

initial_desviation = []
def train_nn (cant_epoch, batch=0):
    desviation_epoch = []
    #para poder reproducir los resultados, limitamos el "random" de las inicializaciones
    np.random.seed(7)
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    for e in cant_epoch:
        model = Sequential()
        model.add(Dense(2, input_dim=2,activation='sigmoid'))
        model.add(Dense(2, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))

        sgd = SGD(lr=0.5)
        model.compile(loss='binary_crossentropy', optimizer=sgd)
        print("entrenando para ", e, " epochs")
        model.fit(X, y, epochs=e)

        print(' ')
        print('####PREDICTION#####')
        print(model.predict_proba(X))
    
        #extraer las activaciones
        ##hidden 1
        model2 = Sequential()
        model2.add(Dense(2, input_dim=2, weights=model.layers[0].get_weights(), activation='sigmoid'))
        activations = model2.predict_proba(X)
        a = np.asarray(activations)
        desviation_one = calc_metric(1, 2, a, y)

        ##hidden 2
        model3 = Sequential()
        model3.add(Dense(2, input_dim=2, weights=model.layers[1].get_weights(), activation='sigmoid'))
        activations2 = model3.predict_proba(activations)
        c = np.asarray(activations2)
        desviation_two = calc_metric(2, 2, c, y)

        ##output
        model4 = Sequential()
        model4.add(Dense(1, input_dim=2, weights=model.layers[2].get_weights(), activation='sigmoid'))
        activations3 = model4.predict_proba(activations2)
        b = np.asarray(activations3)
        desviation_three = calc_metric(3, 1, b, y)
        if(e == 0):
            initial_desviation = np.array([desviation_one, desviation_two, desviation_three])
            print(initial_desviation)
        else:
            desviation_epoch = np.array([desviation_one, desviation_two, desviation_three])
            print(desviation_epoch)
            input('enter para continuar')
