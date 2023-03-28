from keras.callbacks import LearningRateScheduler

def learningrateScheduler(epoch, lr):
    if epoch % 5 == 0:
        return lr/2
    else:
        return lr

learningRateCallback = LearningRateScheduler(learningrateScheduler, verbose=1)
