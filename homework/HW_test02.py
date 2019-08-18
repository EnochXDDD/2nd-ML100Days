import os
import numpy as np
import warnings
from scipy.misc import imresize, imread
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
import tensorflow as tf
warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DATA_PATH = "./data/ml-marathon-final/data/kaggle_dogcat/"
TRAIN_PATH = os.path.abspath(DATA_PATH + "train/")
TEST_PATH = os.path.abspath(DATA_PATH + "test/")
batch_size = 512
num_classes = 2
epochs = 50


def loadTrainData(path):
    images, labels = [], []
    for root, dir, file in os.walk(path):
        for f in file:
            images.append(imresize(imread(os.path.join(root, f)), (64, 64)))
            labels.append(1 if f[:3] == "cat" else 0)
    return np.asarray(images, dtype="float32"), np.asarray(labels)


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(2, activation='sigmoid'))
    return model


x, y = loadTrainData(TRAIN_PATH)
x /= 255.
y = to_categorical(y, num_classes)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
print(x_train.shape, y_train.shape)
print(x_train.dtype, y_train.dtype)

model = define_model()
print(model.summary())

opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

images = []
for i in range(400):
    images.append(imresize(imread(os.path.abspath(TEST_PATH + "/{:03d}.jpg".format(i))), (64, 64)))
images = np.asarray(images, dtype="float32") / 255.
y_pred = model.predict(images)[:, 1]
with open("./result/ml-marathon-final.csv", "w") as f:
    f.write("ID,Predicted\n")
    for i in range(400):
        f.write("{},{}\n".format(i, y_pred[i]))
