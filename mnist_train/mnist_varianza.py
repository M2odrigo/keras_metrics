# imports for array-handling and plotting
import numpy as np
def calc_varianza (r, media, clase, contador):
    sum_tmp = 0
    for num in r:
        print('r : ', num )
        #input('')
        print('vamos a restar cada array en R con la media') 
        print(" ") 
        tmp = np.subtract(media, num) 
        print(tmp)
        print('vamos a elevar al cuadrado cada diferencia')  
        print(" ") 
        tmp = np.square(tmp)
        print(tmp)
        print('vamos a sumar todos los elementos del array')
        print(" ") 
        tmp = np.sum(tmp)
        print(tmp)
        print('vamos a sacar la raiz cuadrada a partir de la suma')
        print(" ") 
        tmp = np.sqrt(tmp)
        print(tmp)
        print('vamos a acumular para el siguiente conjunto de array')
        print(" ") 
        sum_tmp = sum_tmp + tmp
    varianza = 0
    print('sum_tmp', sum_tmp)
    if (sum_tmp == 0):
        return varianza    
    varianza = sum_tmp / contador 
    return varianza

def calc_media (prediction, clase, Y_test):
    cont = 0
    sum_class_zero = np.array([])
    contador_zero=0
    r=np.array([])
    #print('recibimos')
    #print('pred ', prediction)
    #print('ytest', Y_test)
    #input('')
    for index,p in enumerate(prediction):
        cont = cont + 1
        if cont > 50:
            break

        r1 = [format(x, 'f') for x in p]
        r1 = [float(x) for x in p]

        #print(np.asarray(r))
        fields=[r1,str(Y_test[index]), str(Y_test[index].argmax())]
        y = str(Y_test[index].argmax())
        if(str(y)==str(clase)):
            if np.any(r):
                r= np.append(r, [r1], axis=0)
            else:
                r = [r1]

            contador_zero = contador_zero +1
            if np.any(sum_class_zero):
                sum_class_zero = np.sum([sum_class_zero, np.asarray(r1)], axis=0)
            else:
                sum_class_zero = np.asarray(r1)
                #print ("sum", sum_class_zero)
                
    print ("sum", sum_class_zero)
    print ("cont", contador_zero)   
    media = np.divide(sum_class_zero, contador_zero)
    return media, contador_zero

def get_values_classes (prediction, clase, Y_test):
    r=np.array([])
    cont = 0
    for index,p in enumerate(prediction):
        cont = cont + 1
        if cont > 50:
            break

        r1 = [format(x, 'f') for x in p]
        r1 = [float(x) for x in p]

        #print(np.asarray(r))
        fields=[r1,str(Y_test[index]), str(Y_test[index].argmax())]
        y = str(Y_test[index].argmax())
        if(str(y)==str(clase)):
            if np.any(r):
                r= np.append(r, [r1], axis=0)
            else:
                r = [r1]
    return r


