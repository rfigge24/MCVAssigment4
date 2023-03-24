import tensorflow as tf
from keras.layers import LeakyReLU

#-------------------------------------------BaseLine model:
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
    tf.keras.layers.Dropout(0.25)
)
baseModel.add(
    tf.keras.layers.Conv2D(
    filters = 64,
    kernel_size = (3, 3),
    padding = 'valid',
    activation = 'relu'
    )
)
baseModel.add(
    tf.keras.layers.Dropout(0.1)
)
baseModel.add(
    tf.keras.layers.Conv2D(
    filters = 128,
    kernel_size = (3, 3),
    padding = 'valid',
    activation = 'relu'
    )
)
baseModel.add(
    tf.keras.layers.Flatten()
)
baseModel.add(
    tf.keras.layers.Dense(
    128,
    activation = 'relu'
    )
)
baseModel.add(
    tf.keras.layers.Dense(
    64,
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

#-------------------------------------------Variation model 1:
variationModel1 = tf.keras.Sequential()         #val_loss: 0.2318 - val_accuracy: 0.9202

#Adding layers to the base model:
variationModel1.add(
    tf.keras.layers.Conv2D(
    filters = 64,
    kernel_size = (3, 3),
    padding = 'same',
    activation = LeakyReLU(alpha=0.01),
    input_shape = (28, 28, 1)
    )
)
variationModel1.add(
    tf.keras.layers.Conv2D(
    filters = 64,
    kernel_size = (3, 3),
    padding = 'same',
    activation = LeakyReLU(alpha=0.01),
    input_shape = (28, 28, 1)
    )
)
variationModel1.add(
    tf.keras.layers.MaxPooling2D()
)
variationModel1.add(
    tf.keras.layers.Conv2D(
    filters = 128,
    kernel_size = (5, 5),
    padding = 'valid',
    activation = LeakyReLU(alpha=0.01)
    )
)
variationModel1.add(       
    tf.keras.layers.Conv2D(
    filters = 128,
    kernel_size = (5, 5),
    padding = 'valid',
    activation = LeakyReLU(alpha=0.01)
    )
)
variationModel1.add(
    tf.keras.layers.MaxPooling2D()
)
variationModel1.add(
    tf.keras.layers.Flatten()
)
variationModel1.add(
    tf.keras.layers.Dense(
    64,
    activation = LeakyReLU(alpha=0.01)
    )
)
variationModel1.add(
    tf.keras.layers.Dense(
    64,
    activation = LeakyReLU(alpha=0.01)
    )
)
variationModel1.add(
    tf.keras.layers.Dense(
    64,
    activation = LeakyReLU(alpha=0.01)
    )
)
variationModel1.add(
    tf.keras.layers.Dense(
    10,
    activation=tf.keras.activations.softmax
    )
)
#Compiling Base Model:
variationModel1.compile(optimizer= tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])