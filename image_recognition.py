import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
for i in range(10):
	img = x_train[i]
	plt.imshow(img,cmap = "Greys")
	plt.show()


#Normalizing the matrix of features to make the ANN train
#faster. We squize the [0,255] range to [0.01, 1]
x_train = 0.99*x_train.reshape(60000,int(28*28))/255 + 0.01
x_test = 0.99*x_test.reshape(10000,int(28*28))/255 + 0.01

#Encoding categorical variables
y1 = np.zeros((60000,10))
y2 = np.zeros((10000,10))

for i in range(60000):
	y1[i][y_train[i]] = 1

for i in range(10000):
	y2[i][y_test[i]] = 1

#Creating a Neural Network
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim = 16,
	                 init = 'uniform',
	                 activation = 'relu',
	                 input_dim = 28*28))

classifier.add(Dense(output_dim = 16,
	                 init = 'uniform',
	                 activation = 'relu'))

classifier.add(Dense(output_dim = 10,
	                 init = 'uniform',
	                 activation = 'sigmoid'))

classifier.compile(optimizer = 'adam',
	               loss = 'binary_crossentropy',
	               metrics = ['accuracy'])

classifier.fit(x_train,y1,
	           batch_size = 600,
	           nb_epoch = 100)

y_pred = classifier.predict(x_test)

#Turning probabilities to the defenite results
for i in range(10000):
	b = y_pred[i][0]
	k = 0
	for j in range(1,10):
		if y_pred[i][j]>b:
			b = y_pred[i][j]
			k = j
	y_pred[i] = 0
	y_pred[i][k] = 1

y3 = np.zeros(10000)

for i in range(10000):
	for j in range(10):
		if y_pred[i][j] == 1:
			y3[i] = int(j)

#Cheking the results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y3)
print(cm)

sum = 0
for i in range(10):
	sum = sum+cm[i][i]

print('Accuracy of the training set: ', sum/10000)
print('Number of incorrectly predicted values: ', 10000-sum)

y_incorrect = np.zeros((10000-sum,3))
k = 0

for i in range(10000):
	if y_test[i] != y3[i]:
		y_incorrect[k] = [y_test[i],y3[i],i]
		k = k+1

#Visualizing incorrectly guessed answers
print("If you want to see the incorrectly predicted results press (ENTER), otherwise type (exit)")
for i in range(10000-sum):
	ask = raw_input()
	if ask == 'exit':
		break
	else:
		img = x_test[int(y_incorrect[i][2])].reshape((28,28))
		print("Real Value: %d  Predicted Value: %d" %(y_incorrect[i][0],y_incorrect[i][1]))
		plt.imshow(img,cmap = "Greys")
		plt.show()
