import tensorflow as tf
from keras.layers import LeakyReLU

#-------------------------------------model2
baseModel = tf.keras.Sequential()         #val_loss: 0.2572 - val_accuracy: 0.9137

#Adding layers to the base model:
baseModel.add(
    tf.keras.layers.Conv2D(
    filters = 6,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    input_shape = (28, 28, 1)
    )
)
baseModel.add(
    tf.keras.layers.MaxPooling2D()
)
baseModel.add(
    tf.keras.layers.Conv2D(
    filters = 16,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    )
)
baseModel.add(
    tf.keras.layers.MaxPooling2D()
)
baseModel.add(
    tf.keras.layers.Conv2D(
    filters = 32,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    )
)
baseModel.add(
    tf.keras.layers.Flatten()
)
baseModel.add(
    tf.keras.layers.Dense(
    64,
    activation = 'relu'
    )
)
baseModel.add(
    tf.keras.layers.Dense(
    120,
    activation = 'relu'
    )
)
baseModel.add(
    tf.keras.layers.Dense(
    80,
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
baseModel.compile(optimizer= tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])