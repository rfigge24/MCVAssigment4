from keras.datasets import fashion_mnist

import os
import numpy as np
import shutil
import tensorflow as tf

import visualizationPlotting as visplot
import variantModels

#loading the data and setting up the label list:
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#normalizing the images: to avoid to large output values that can produce a vanishing gradientl
x_train = x_train / 255.0
x_test = x_test / 255.0

def main(models):
    for model,modelname, nrOfEpochs in models:
        print(f'Evaluating model {modelname}.')
        model.summary()

        outPath = modelname
        #Create a directory to save the model plots and its performances:
        if not os.path.exists(outPath):
            print("Model not trained!")
            
        # load the best weights from the saved file
        model.load_weights(f'{modelname}/best_performing_weights.h5')

        print("Evaluating...")
        results = model.evaluate(x_test, y_test)
        print("Test loss, test accuracy: ", results)

if __name__ == '__main__':
    modelList = [
        (variantModels.dropoutModel,    'Dropout Model', 20),
        (variantModels.poolingModel,    'Extra Pooling Model', 10)
    ]
    main(modelList)