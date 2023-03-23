from keras.datasets import fashion_mnist
import tensorflow as tf
from sklearn.model_selection import train_test_split


#loading the data and setting up the label list:
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#splitting the training data into a training and validation set with a 8 to 2 ratio, while keeping the same class distribution be using stratify:
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train,random_state = 0, stratify=y_train, test_size=0.20)

#normalizing the images: for computational convienence
x_train = x_train / 255.0
x_validate = x_validate / 255.0
x_test = x_test / 255.0

#BaseLine model:
baseModel = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

baseModel.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
callback = baseModel.fit(x_train, y_train, epochs=3)
print(callback.history.keys())


#Variation models:


