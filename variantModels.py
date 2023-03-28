import tensorflow as tf
from keras.layers import LeakyReLU

# -------- Dropout Model --------
dropoutModel = tf.keras.Sequential()

#Adding layers to the dropout model:
dropoutModel.add(
    tf.keras.layers.Conv2D(
    filters = 6,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    input_shape = (28, 28, 1)
    )
)
dropoutModel.add(
    tf.keras.layers.MaxPooling2D()
)
dropoutModel.add(
    tf.keras.layers.Dropout(0.3)
)
dropoutModel.add(
    tf.keras.layers.Conv2D(
    filters = 16,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    )
)
dropoutModel.add(
    tf.keras.layers.MaxPooling2D()
)
dropoutModel.add(
    tf.keras.layers.Dropout(0.1)
)
dropoutModel.add(
    tf.keras.layers.Conv2D(
    filters = 32,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    )
)
dropoutModel.add(
    tf.keras.layers.Flatten()
)
dropoutModel.add(
    tf.keras.layers.Dense(
    64,
    activation = 'relu'
    )
)
dropoutModel.add(
    tf.keras.layers.Dense(
    120,
    activation = 'relu'
    )
)
dropoutModel.add(
    tf.keras.layers.Dense(
    80,
    activation = 'relu'
    )
)
dropoutModel.add(
    tf.keras.layers.Dense(
    10,
    activation=tf.keras.activations.softmax
    )
)
#Compiling Base Model:
dropoutModel.compile(optimizer= tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# -------- generalisation model --------
batchNormModel = tf.keras.Sequential()

#Adding layers to the base model:
batchNormModel.add(
    tf.keras.layers.Conv2D(
    filters = 6,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    input_shape = (28, 28, 1)
    )
)
batchNormModel.add(
    tf.keras.layers.BatchNormalization()
)
batchNormModel.add(
    tf.keras.layers.MaxPooling2D()
)
batchNormModel.add(
    tf.keras.layers.Conv2D(
    filters = 16,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    )
)
batchNormModel.add(
    tf.keras.layers.MaxPooling2D()
)
batchNormModel.add(
    tf.keras.layers.Conv2D(
    filters = 32,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    )
)
batchNormModel.add(
    tf.keras.layers.Flatten()
)
batchNormModel.add(
    tf.keras.layers.Dense(
    64,
    activation = 'relu'
    )
)
batchNormModel.add(
    tf.keras.layers.Dense(
    120,
    activation = 'relu'
    )
)
batchNormModel.add(
    tf.keras.layers.Dense(
    80,
    activation = 'relu'
    )
)
batchNormModel.add(
    tf.keras.layers.Dense(
    10,
    activation=tf.keras.activations.softmax
    )
)
#Compiling Base Model:
batchNormModel.compile(optimizer= tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# -------- Reshaped dense layers --------
denseModel = tf.keras.Sequential()         #val_loss: 0.2572 - val_accuracy: 0.9137

#Adding layers to the base model:
denseModel.add(
    tf.keras.layers.Conv2D(
    filters = 6,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    input_shape = (28, 28, 1)
    )
)
denseModel.add(
    tf.keras.layers.MaxPooling2D()
)
denseModel.add(
    tf.keras.layers.Conv2D(
    filters = 16,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    )
)
denseModel.add(
    tf.keras.layers.MaxPooling2D()
)
denseModel.add(
    tf.keras.layers.Conv2D(
    filters = 32,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    )
)
denseModel.add(
    tf.keras.layers.Flatten()
)
denseModel.add(
    tf.keras.layers.Dense(
    128,
    activation = 'relu'
    )
)
denseModel.add(
    tf.keras.layers.Dense(
    64,
    activation = 'relu'
    )
)
denseModel.add(
    tf.keras.layers.Dense(
    32,
    activation = 'relu'
    )
)
denseModel.add(
    tf.keras.layers.Dense(
    10,
    activation=tf.keras.activations.softmax
    )
)
#Compiling Base Model:
denseModel.compile(optimizer= tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# -------- Extra pooling --------
poolingModel = tf.keras.Sequential()         #val_loss: 0.2572 - val_accuracy: 0.9137

#Adding layers to the base model:
poolingModel.add(
    tf.keras.layers.Conv2D(
    filters = 6,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    input_shape = (28, 28, 1)
    )
)
poolingModel.add(
    tf.keras.layers.MaxPooling2D()
)
poolingModel.add(
    tf.keras.layers.Conv2D(
    filters = 16,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    )
)
poolingModel.add(
    tf.keras.layers.MaxPooling2D()
)
poolingModel.add(
    tf.keras.layers.Conv2D(
    filters = 32,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    )
)
poolingModel.add(
    tf.keras.layers.MaxPooling2D()
)
poolingModel.add(
    tf.keras.layers.Flatten()
)
poolingModel.add(
    tf.keras.layers.Dense(
    64,
    activation = 'relu'
    )
)
poolingModel.add(
    tf.keras.layers.Dense(
    120,
    activation = 'relu'
    )
)
poolingModel.add(
    tf.keras.layers.Dense(
    80,
    activation = 'relu'
    )
)
poolingModel.add(
    tf.keras.layers.Dense(
    10,
    activation=tf.keras.activations.softmax
    )
)
#Compiling Base Model:
poolingModel.compile(optimizer= tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# -------- base model --------
smallKernelModel = tf.keras.Sequential()         #val_loss: 0.2572 - val_accuracy: 0.9137

#Adding layers to the base model:
smallKernelModel.add(
    tf.keras.layers.Conv2D(
    filters = 6,
    kernel_size = 3,
    padding = 'same',
    activation = 'relu',
    input_shape = (28, 28, 1)
    )
)
smallKernelModel.add(
    tf.keras.layers.MaxPooling2D()
)
smallKernelModel.add(
    tf.keras.layers.Conv2D(
    filters = 16,
    kernel_size = 3,
    padding = 'same',
    activation = 'relu',
    )
)
smallKernelModel.add(
    tf.keras.layers.MaxPooling2D()
)
smallKernelModel.add(
    tf.keras.layers.Conv2D(
    filters = 32,
    kernel_size = 3,
    padding = 'same',
    activation = 'relu',
    )
)
smallKernelModel.add(
    tf.keras.layers.Flatten()
)
smallKernelModel.add(
    tf.keras.layers.Dense(
    64,
    activation = 'relu'
    )
)
smallKernelModel.add(
    tf.keras.layers.Dense(
    120,
    activation = 'relu'
    )
)
smallKernelModel.add(
    tf.keras.layers.Dense(
    80,
    activation = 'relu'
    )
)
smallKernelModel.add(
    tf.keras.layers.Dense(
    10,
    activation=tf.keras.activations.softmax
    )
)
#Compiling Base Model:
smallKernelModel.compile(optimizer= tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])