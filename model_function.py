# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 10:38:19 2019

@author: NISARG
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import CSVLogger


IMAGE_SIZE = 200
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS = 20
BATCH_SIZE = 32
TEST_SIZE = 30
input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

model = Sequential()

model.add(Conv2D(64, 3, 3, border_mode='same', input_shape=input_shape, activation='relu'))
model.add(ZeroPadding2D(padding=((1,2),(3,4))))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(256, activation='relu'))


model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(ZeroPadding2D(padding=((1,2),(3,4))))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(256, activation='relu'))


model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(ZeroPadding2D(padding=((1,2),(3,4))))
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(256, activation='relu'))


model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(ZeroPadding2D(padding=((1,2),(3,4))))
model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(256, activation='relu'))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(1))
model.add(Activation('sigmoid'))
    
model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(lr=0.0001),
            metrics=['accuracy'])