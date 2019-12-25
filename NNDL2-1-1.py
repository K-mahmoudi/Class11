import keras
from keras.models import Sequential
from keras.layers import Dense
from mlxtend.data import loadlocal_mnist
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

batch_size = 100
num_classes = 10
epochs = 300


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    x_train, y_train = loadlocal_mnist(
        images_path='samples/train-images-idx3-ubyte',
        labels_path='samples/train-labels-idx1-ubyte')

    x_test, test_label = loadlocal_mnist(
        images_path='samples/t10k-images-idx3-ubyte',
        labels_path='samples/t10k-labels-idx1-ubyte')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(test_label, num_classes)

    x_test = x_test.astype(np.float32)
    x_train = x_train.astype(np.float32)
    x_test[:][:] = x_test[:][:] / 255
    x_train[:][:] = x_train[:][:] / 255

    y_test[:][:] = y_test[:][:] * 2 - 1
    y_train[:][:] = y_train[:][:] * 2 - 1

    model = Sequential()
    model.add(Dense(512, activation=keras.backend.tanh, input_shape=(784,)))
    model.add(Dense(num_classes, activation=keras.backend.tanh))

    model.summary()

    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.SGD(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test,
                                                                                                            y_test))
    predict = model.predict(x_test, batch_size, 1)

    amx = np.zeros(10000, np.int32)
    for i in range(10000):
        amx = np.argmax(predict, axis=1)
    cnf_matrix = confusion_matrix(test_label, amx)
    ccr = 0
    for c in range(num_classes):
        ccr += cnf_matrix[c][c]
    print(ccr/10000)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                          title='Confusion matrix, without normalization')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    pass


if __name__ == "__main__": main()
