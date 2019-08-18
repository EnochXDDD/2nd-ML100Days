import os
import numpy as np
import warnings
from scipy.misc import imresize, imread
from keras.utils import to_categorical
# import keras
# from keras.preprocessing.image import ImageDataGenerator
# from .resnet_builder import resnet
# import tensorflow as tf
warnings.filterwarnings("ignore")
# tf.logging.set_verbosity(tf.logging.ERROR)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DATA_PATH = "./data/ml-marathon-final/data/kaggle_dogcat/"
TRAIN_PATH = os.path.abspath(DATA_PATH + "train/")
TEST_PATH = os.path.abspath(DATA_PATH + "test/")
batch_size = 2048
num_classes = 2
epochs = 10


def loadTrainData(path):
    images, labels = [], []
    for root, dir, file in os.walk(path):
        for f in file:
            images.append(imresize(imread(os.path.join(root, f)), (64, 64)))
            labels.append(1 if f[:3] == "cat" else 0)
    return np.asarray(images), np.asarray(labels)


x_train, y_train = loadTrainData(TRAIN_PATH)
y_train = to_categorical(y_train, num_classes)
print(x_train.shape, y_train.shape)
print(x_train.dtype, y_train.dtype)

# model = resnet(input_shape=(32,32,3))
# model.summary()