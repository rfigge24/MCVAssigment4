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
#Compiling Dropout Model:
dropoutModel.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# -------- Extra pooling --------
poolingModel = tf.keras.Sequential()

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
#Compiling Pooling Model:
poolingModel.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# -------- Smaller Kernels Model --------
smallKernelModel = tf.keras.Sequential()

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
#Compiling Small kernel Model:
smallKernelModel.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# -------- less dense model --------
lessDense = tf.keras.Sequential()         #val_loss: 0.2572 - val_accuracy: 0.9137

#Adding layers to the base model:
lessDense.add(
    tf.keras.layers.Conv2D(
    filters = 6,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    input_shape = (28, 28, 1)
    )
)
lessDense.add(
    tf.keras.layers.MaxPooling2D()
)
lessDense.add(
    tf.keras.layers.Conv2D(
    filters = 16,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    )
)
lessDense.add(
    tf.keras.layers.MaxPooling2D()
)
lessDense.add(
    tf.keras.layers.Conv2D(
    filters = 32,
    kernel_size = 5,
    padding = 'same',
    activation = 'relu',
    )
)
lessDense.add(
    tf.keras.layers.Flatten()
)
lessDense.add(
    tf.keras.layers.Dense(
    64,
    activation = 'relu'
    )
)
lessDense.add(
    tf.keras.layers.Dense(
    80,
    activation = 'relu'
    )
)
lessDense.add(
    tf.keras.layers.Dense(
    10,
    activation=tf.keras.activations.softmax
    )
)
#Compiling Less Dense Model:
lessDense.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
