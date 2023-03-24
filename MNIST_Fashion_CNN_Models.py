from keras.datasets import fashion_mnist
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import visualizationPlotting as visplot

#loading the data and setting up the label list:
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#splitting the training data into a training and validation set with a 8 to 2 ratio, while keeping the same class distribution be using stratify:
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train,random_state = 0, stratify=y_train, test_size=0.20)



#normalizing the images: to avoid to large output values that can produce a vanishing gradientl
x_train = x_train / 255.0
x_validate = x_validate / 255.0
x_test = x_test / 255.0


#viewing the data:
if False:
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i]])
    plt.show()

#BaseLine model:
baseModel = tf.keras.Sequential()

#Adding layers to the base model:
baseModel.add(
    tf.keras.layers.Conv2D(
    filters = 32,
    kernel_size = (3, 3),
    padding = 'valid',
    activation = 'relu',
    input_shape = (28, 28, 1)
    )
)

baseModel.add(
    tf.keras.layers.MaxPool2D()
)

baseModel.add(
    tf.keras.layers.Conv2D(
    filters = 32,
    kernel_size = (3, 3),
    padding = 'valid',
    activation = 'relu'
    )
)

baseModel.add(
    tf.keras.layers.Flatten(input_shape=(14,14))
)

baseModel.add(
    tf.keras.layers.Dense(
    128,
    activation = 'relu'
    )
)

baseModel.add(
    tf.keras.layers.Dense(
    10,
    activation=tf.keras.activations.softmax
    )
)


#Compiling Base Model:
baseModel.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

callback = baseModel.fit(x_train, y_train, validation_data = (x_validate, y_validate), epochs=15)

visplot.plotPerformance(callback)


print(callback.history.keys())


#Variation models:


