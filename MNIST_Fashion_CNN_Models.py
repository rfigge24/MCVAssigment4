from keras.datasets import fashion_mnist
#loading the data:
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#splitting the training data into a training and validation set with a 8 to 2 ratio:
#We want the validation set to generalize over the whole data, so we want the same distribution of classes in our validation set, 
#since the class is uniformly distributed we will take 1200 of each class and put them into our validationset:


