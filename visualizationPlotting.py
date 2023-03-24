from matplotlib import pyplot as plt

def plotPerformance(callback):
    loss = callback.history['loss']
    val_loss = callback.history['val_loss']
    acc = callback.history['accuracy']
    val_acc = callback.history['val_accuracy']
    epochs = range(1,len(loss) + 1)

    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.figure()

    plt.plot(epochs, acc, label='Training accuracy')
    plt.plot(epochs, val_acc, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()