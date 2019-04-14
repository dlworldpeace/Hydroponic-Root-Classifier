import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm, tqdm_notebook
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
# from keras.applications import VGG16


class VGG16:
    @staticmethod
    def build(width, height, depth, classes, reg):

        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(32, (5, 5), strides = (1, 1), name = 'conv0', input_shape = inputShape))

        model.add(BatchNormalization(axis = 3, name = 'bn0'))
        model.add(Activation('relu'))

        model.add(MaxPooling2D((2, 2), name='max_pool'))
        model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
        model.add(Activation('relu'))
        model.add(AveragePooling2D((3, 3), name='avg_pool'))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(300, activation="relu", name='rl'))
        model.add(Dropout(0.5))
        model.add(Dense(1,activation='sigmoid', name='sm'))

		# return the constructed network architecture
        return model