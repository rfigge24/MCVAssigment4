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
        print(f'Training model {modelname} on full trainingset and evaluating on the test set.')
        model.summary()

        outPath = modelname
        #Create a directory to save the model plots and its performances:
        if not os.path.exists(outPath):
            print("Model not trained!")

        #Fitting the model:
        model.fit(x_train, y_train, epochs=nrOfEpochs)
        model.save_weights(f'{modelname}/weights_fulltrainingset.h5')

        print("Evaluating...")
        trainingPerformance = model.evaluate(x_train,y_train)
        testPerformance = model.evaluate(x_test, y_test)
        print("Training loss, Training accuracy: ", trainingPerformance)
        print("Test loss, Test accuracy: ", testPerformance)

        #Confusionmatrix stuff CHOICETASK:
        y_preds = []
        predictions = model.predict(x_test, verbose = 1)
        for i, p in enumerate(predictions):
            y_preds.append(np.argmax(p))

        confusion = tf.math.confusion_matrix(y_test, y_preds)
        visplot.plotConfusionMatrix(confusion, class_names, modelname)

if __name__ == '__main__':
    modelList = [
        (variantModels.dropoutModel,    'Dropout Model', 13),
        (variantModels.poolingModel,    'Extra Pooling Model', 10)
    ]
    main(modelList)