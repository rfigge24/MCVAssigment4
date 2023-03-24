from matplotlib import pyplot as plt

def plotPerformance(callback, modelname):
    loss = callback.history['loss']
    val_loss = callback.history['val_loss']
    acc = callback.history['accuracy']
    val_acc = callback.history['val_accuracy']
    epochs = range(1,len(loss) + 1)

    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title(f'Training and validation loss for {modelname}')
    plt.legend()
    plt.savefig(f'{modelname}/LossPlot.png')
    plt.figure()

    plt.plot(epochs, acc, label='Training accuracy')
    plt.plot(epochs, val_acc, label='Validation accuracy')
    plt.title(f'Training and validation accuracy for {modelname}')
    plt.legend()
    plt.savefig(f'{modelname}/AccuracyPlot.png')
    plt.show()
  