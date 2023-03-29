from keras.callbacks import LearningRateScheduler

def learningrateScheduler(epoch, lr):
    if (epoch+1) % 5 == 0:    #plus 1 since tensorflow seems to keep track of the epochs counting from 0 internally instead of from 1
        return lr/2
    else:
        return lr

learningRateCallback = LearningRateScheduler(learningrateScheduler, verbose=1)
