import scipy.io as io
import h5py
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Conv2D,Dropout,Cropping2D,Convolution2D ,BatchNormalization,MaxPooling2D
from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


if not os.path.isdir("trainedModel"):
    os.mkdir("trainedModel")

test_size=1500
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 140 , 320 , 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
keep_prob=0.7

batch_size=1
samples_per_epoch=32
nb_epoch=1000
data_dir=os.path.join(os.getcwd(),"img")



def load_data():
    data_df = pd.read_csv(os.path.join(data_dir, 'out.csv'))
    X = data_df[['center']].values
    X1 = data_df[['command']].values
    y = data_df[['steer','throttle','brake']].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, X1, y, test_size=test_size, random_state=0)
    return X_train, X_valid, y_train, y_valid


def build_model():
    model1 = Sequential()
    model1.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model1.add(Conv2D(24, 5,strides=(2, 2), activation='elu'))
    model1.add(Conv2D(36, 5, strides=(2, 2), activation='elu'))
    model1.add(Conv2D(48, 5, strides=(2, 2), activation='elu'))
    model1.add(Conv2D(64, 3, activation='elu'))
    model1.add(Conv2D(64, 3, activation='elu'))
    model1.add(Dropout(keep_prob))
    model1.add(Flatten())
    model1.add(Dense(100, activation='elu'))
    
    
    model2 = Sequential()
    model2.add(Dense(1, input_shape=(X1), activation='elu'))
    
    model =  Concatenate([model1, model2])
    model.add(Dense(3, activation='elu'))

    
    #model.add(Dense(50, activation='elu'))
    #model.add(Dense(10, activation='elu'))
    #model.add(Dense(3))
    # model.summary()
    # model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
    
    return model


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    im = image[100: , :, :] #Crop
    #image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA) #Resize
    im = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
    return im

def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty([batch_size,3])
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center = image_paths[index]
            
            steering_angle,throttle,brake= steering_angles[index]
            
            # argumentation
            # if is_training and np.random.rand() < 0.6:
            #     image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            # else:
            image = load_image(data_dir, *center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = [steering_angle,throttle,brake]
            i += 1
            if i == batch_size:
                break
        yield images, steers


def train_model(model, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('trainedModel\\model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only='true',
                                 mode='auto')
                                 
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001),metrics=['accuracy'])
    history=model.fit_generator(batch_generator(data_dir, X_train, y_train, batch_size, True),
                        samples_per_epoch,
                        nb_epoch,
                        max_queue_size=1,
                        validation_data=batch_generator(data_dir, X_valid, y_valid, batch_size, False),
                        validation_steps=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)
    plot1 = plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plot2 = plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
data = load_data()
print("Data  Loded")
model = build_model()
print("Model Loaded")
train_model(model, *data)
print("Model Trained")