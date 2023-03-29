from keras.datasets import fashion_mnist
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
import visualizationPlotting as visplot
from keras.utils.vis_utils import plot_model
import os
import numpy as np
import shutil

import baseModel
import variantModels

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




def main(models, nrOfEpochs, save = True):
    #clear the existing performance text file:
    if os.path.exists('performance.txt'):
        with open('performance.txt', 'w') as file:
            file.truncate(0)


    for model,modelname in models:

        # allow user to skip fitting existing model
        if os.path.exists(modelname):
            if input(f'[{modelname}] already exists. Refit model? Y/N: ').lower()[0] != 'y':
                continue

        print(f'Training model {modelname}.')
        model.summary()

        #Create a directory to save the model plots and its performances:
        if not os.path.exists(modelname):
            os.makedirs(modelname)
        else:
            shutil.rmtree(modelname)

        #Use a ModelCheckpoint callback to keep track of the best performing weights and save them:
        checkpoint = ModelCheckpoint(f'{modelname}/best_performing_weights.h5', monitor='val_accuracy', save_best_only=True)

        #Fitting the model:
        history = model.fit(x_train, y_train, validation_data = (x_validate, y_validate), epochs=nrOfEpochs, callbacks = [checkpoint])

        # load the best weights from the saved file
        model.load_weights(f'{modelname}/best_performing_weights.h5')

        #get the epoch with the max validation accuracy and the corresponding performance values:
        epoch = np.argmax(history.history["val_accuracy"])
        validationString = f'validation Loss: {history.history["val_loss"][epoch]}, validation Accuracy: {history.history["val_accuracy"][epoch]}'
        trainingString = f'training Loss: {history.history["loss"][epoch]}, training Accuracy: {history.history["accuracy"][epoch]}'

        print(trainingString)
        print(validationString)

        #write the performance to a textfile
        with open('performance.txt','a', encoding="UTF-8") as file:
            file.write(f'{modelname}:\n')
            file.write(trainingString + '\n')
            file.write(validationString + '\n')

        #plot the performance:
        visplot.plotPerformance(history, modelname, nrOfEpochs)
        
        #plot a grapical scheme of the network
        plot_model(model, to_file=f'{modelname}/Network_Graph.png', show_shapes=True, show_layer_names=True, show_layer_activations = True)

if __name__ == '__main__':
    modelList = [
        (baseModel.baseModel,           'Base Model'),
        (variantModels.dropoutModel,    'Dropout Model'),
        (variantModels.batchNormModel,  'Normalization Model'),
        (variantModels.denseModel,      'Reshaped Dense Model'),
        (variantModels.poolingModel,    'Extra Pooling Model'),
        (variantModels.smallKernelModel,'Smaller Kernel Model'),
        (variantModels.learningRate,    'higher learningRate Model'),
        (variantModels.lessDense,       'One Less Dense Layer Model')
    ]
    main(modelList, 2)