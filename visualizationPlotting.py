from matplotlib import pyplot as plt
import os

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
