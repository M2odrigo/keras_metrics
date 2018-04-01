import numpy as np
##funcion para calcular la desviacion
def calc_metric(capa, nodos, activaciones, y):
    print("capa ", capa)
    print("nodos", nodos)
    print("y ", y.shape)
    print("activaciones ", activaciones.shape)
    desviation = []
    h2 = []
    sum_zero = [np.zeros(nodos,)]
    sum_one = [np.zeros(nodos,)]
    cont_zero = 0
    cont_one = 0
    class_zero = 0
    class_one = 0
    for index, w in enumerate(activaciones):
        h2.append(np.append([activaciones[index]], [y[index]]))
        if(y[index] == 1):
            cont_one = cont_one + 1
            sum_one = np.sum([sum_one, activaciones[index]], axis=0)
            #class_one = np.append
        else:
            cont_zero = cont_zero + 1
            sum_zero = np.sum([sum_zero, activaciones[index]], axis=0)

    print("cant zeros", cont_zero)
    print("cant ones", cont_one)
    #dividimos para hallar el promedio
    sum_zero[:] = [x / cont_zero for x in sum_zero]
    sum_one[:] = [x / cont_one for x in sum_one]

    #restar cada output con su media y elevar al cuadrado
    for index, w in enumerate(activaciones):
        h2.append(np.append([activaciones[index]], [y[index]]))
        if(y[index] == 1):
            #print("a2[index]",activaciones[index])
            #print("sum_one ", sum_one)
            calc = activaciones[index] - sum_one
            #print("se ha restado el promedio", calc)
            calc = np.power(calc,2)
            #print("se ha elevado al cuadrado", calc)
            partial_sum = np.sum(calc)
            #print("partial sum", partial_sum)
            class_one = class_one + partial_sum
            #print("class one content", class_one)
            #input('')

        else:
            #print("a2[index]",activaciones[index])
            #print("sum_one ", sum_zero)
            calc = activaciones[index] - sum_zero
            #print("se ha restado el promedio", calc)
            calc = np.power(calc,2)
            #print("se ha elevado al cuadrado", calc)
            partial_sum = np.sum(calc)
            #print("partial sum", partial_sum)
            class_zero = class_zero + partial_sum
            #print("class one content", class_zero)
            #input('')
                
    class_one = float(class_one / cont_one)
    class_zero = float(class_zero / cont_zero)
    print(capa, " Layer - Desviation class one: ", class_one)
    print(capa, " Layer - Desviation class zero: ", class_zero)
    desviation = np.array([[class_zero], [class_one]])
    #se devuelven las desviaciones por clase
    desviation_zero = np.array([class_zero])
    desviation_one = np.array([class_one])
    return desviation_zero, desviation_one
