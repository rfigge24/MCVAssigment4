from keras.datasets import fashion_mnist
import tensorflow as tf
from sklearn.model_selection import train_test_split
import visualizationPlotting as visplot
from keras.utils.vis_utils import plot_model
import os
import glob

import models

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




def main(models):
    for model,modelname in models:
        #Fitting the model:
        callback = model.fit(x_train, y_train, validation_data = (x_validate, y_validate), epochs=1)
        
        

        #Create a directory to save the model plots and its performances:
        if not os.path.exists(modelname):
            os.makedirs(modelname)
        else:
            files = glob.glob(modelname + '/*')
            for file in files:
                os.remove(file)

        #plot the performance:
        visplot.plotPerformance(callback, modelname)

        #plot a grapical scheme of the network
        plot_model(model, to_file=f'{modelname}/Network_Graph.png', show_shapes=True, show_layer_names=True)



if __name__ == '__main__':
    modelList = [
        (models.baseModel,'Base Model')
    ]
    
    main(modelList)