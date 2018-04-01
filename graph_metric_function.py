import numpy as np
import matplotlib.pyplot as plt

#X --> layers
#Y --> valores de las metricas
#Epoch --> array con los epoch de cada metrica
def graph_metric(x,y, epoch):
    i = 0
    for l in x:
        #sirve para aumentar el subplot y se vean una sola vista los diferentes graficos
        #por cada layer, significa una clase diferente
        clase = i
        i = i+1
        print('layer : ', i, ' values ', l)
        #se recibiran N conjuntos de metricas, la primera es la clase 0, se accede por el indice i sin el aumento ++
        for index, m in enumerate(y[clase]):
            label_epoch = 'epoch {0}'.format(epoch[index+1])
            print(label_epoch)
            print("x", l)
            print("y", m)
            #solamente en una figura, si se tiene que separar las figuras por clase, asignar plt.figure(i)
            #plt.figure(1)
            #numRows, numCols, figure (211) --- (212)
            plt.subplot(210+i)
            #titulo del graph
            plt.title('Class {0}'.format(clase))
            #titulos de los ejes
            plt.ylabel('metricas',color='red')
            plt.xlabel('layers',color='red')
            #establecer los valores para los ejes plt.axis([xmin,xmax,ymin,ymax])
            #plt.axis([0, 3, 0, 60])
            plt.plot(l, m, label=label_epoch)

    plt.legend()
    plt.show()
