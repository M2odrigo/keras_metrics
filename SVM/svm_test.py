#Import Library
from sklearn import svm
import numpy
#Assumed you have, X (predictor) and Y (target) 
#for training data set and x_test(predictor) of test_dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
y = dataset[:,8]
# Create SVM classification object 
model = svm.SVC(kernel='linear', C=1, gamma=1) 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(X)
