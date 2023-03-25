from matplotlib import pyplot as plt
import os
import numpy as np

def plotPerformance(history, modelname, nrOfSetEpochs):
    
    if not os.path.exists(modelname):
            os.makedirs(modelname)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1,len(loss) + 1)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xlim(1,nrOfSetEpochs)

    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title(f'Training and validation loss for {modelname}')
    plt.legend()
    plt.savefig(f'{modelname}/LossPlot.png')
    plt.figure()

    plt.ylabel("Accuracy")

    plt.plot(epochs, acc, label='Training accuracy')
    plt.plot(epochs, val_acc, label='Validation accuracy')
    plt.title(f'Training and validation accuracy for {modelname}')
    plt.legend()
    plt.savefig(f'{modelname}/AccuracyPlot.png')
    plt.show()



def plotConfusionMatrix(Labels, Predictions, nrOfClasses, labelNames, modelname):
    if len(Labels) != len(Predictions) != len(labelNames): return
    
	#build the matrix: Rows are True values and the Columns are the predictions
    matrix = np.zeros((nrOfClasses,nrOfClasses))
    for label, prediction in zip (Labels,Predictions):
        matrix[label,prediction] += 1
    
    #plot the matrix:
    plt.figure(figsize=(12,10))
    c = plt.imshow(matrix,cmap ='Blues',vmin = np.amin(matrix) ,vmax = np.amax(matrix))
    plt.colorbar(c,)

    plt.xticks(np.arange(nrOfClasses), labels = labelNames)
    plt.yticks(np.arange(nrOfClasses), labels = labelNames)

    plt.ylabel("True Labels:", fontsize = 20)
    plt.xlabel("predicted Labels", fontsize = 20)

    for i in range(nrOfClasses):
         for j in range(nrOfClasses):
              plt.text(i, j , matrix[j][i], ha="center", va="center", color="black", fontweight ="bold")

    plt.savefig(f'{modelname}/ConfusionmatrixPlot.png')
    plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plotConfusionMatrix([1,2,1,4,3,5,6,4,6,7,4,3,4,5,9,0,8,6,4,3,2,4,5,7,8,8,5,4,2],[2,2,1,4,3,5,6,4,6,7,4,3,4,5,9,0,8,6,4,3,2,4,5,7,8,8,5,4,2],10, class_names)

      