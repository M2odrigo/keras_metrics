# Create your first MLP in Keras
from desviation_function import calc_metric
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
# fix random seed for reproducibility
np.random.seed(7)

##variables
cant_epoch = 0
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
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=cant_epoch, batch_size=batch_size)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print(model.summary())

print("#########outputs of hidden layer 1 with 12 nodes#################")
model2 = Sequential()
model2.add(Dense(12, input_dim=8, weights=model.layers[0].get_weights(), activation='sigmoid'))
activations2 = model2.predict_proba(X)
#print(activations2)
a2 = np.asarray(activations2)
desviation_one = calc_metric(1, 12, a2, Y)
print("desviation_one",desviation_one)
print("desviation_class0e",desviation_one[0])
print("desviation_class1e",desviation_one[1])

#np.savetxt(str(cant_epoch) + "_epoch_hidden1.csv", h2, fmt = '%.5f', delimiter=",")

print("#########outputs of hidden layer 2 with 8 nodes#################")
model3 = Sequential()
model3.add(Dense(8, input_dim=12, weights=model.layers[1].get_weights(), activation='sigmoid'))
activations3 = model3.predict_proba(activations2)
#print(activations3)
a3 = np.asarray(activations3)
desviation_two = calc_metric(2, 8, a3, Y)
print("desviation_two",desviation_two)
print("desviation_class0e",desviation_two[0])
print("desviation_class1e",desviation_two[1])
#h3 = []
#for index, w in enumerate(a3):
#    h3.append(np.append([a3[index]], [Y[index]]))
#np.savetxt(str(cant_epoch) + "_epoch_hidden2.csv", h3, fmt = '%.5f', delimiter=",")

print("#########outputs of output layer with 1 node#################")
model4 = Sequential()
model4.add(Dense(1, input_dim=8, weights=model.layers[2].get_weights(), activation='sigmoid'))
activations4 = model4.predict_proba(activations3)
#print(activations4)
a4 = np.asarray(activations4)
desviation_three = calc_metric(3, 1, a4, Y)
print("desviation_three",desviation_three)
print("desviation_class0e",desviation_three[0])
print("desviation_class1e",desviation_three[1])
#h4 = []
#for index, w in enumerate(a4):
#    h4.append(np.append([a4[index]], [Y[index]]))
#np.savetxt(str(cant_epoch) + "_epoch_output.csv", h4, fmt = '%.5f', delimiter=",")
