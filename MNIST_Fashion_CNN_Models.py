from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import pandas as pd
#loading the data:
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#splitting the training data into a training and validation set with a 8 to 2 ratio, while keeping the same class distribution be using stratify:
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train,random_state = 0, stratify=y_train, test_size=0.20)




