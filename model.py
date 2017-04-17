import os
import csv

#Read data samples from the CSV file
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples = samples[1:]   #remove the headings line

# split data samples randomly into 80% training an 20% validation
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("Image data is split into {} training and {} validation samples.".format(len(train_samples), len(validation_samples)))

import cv2
import numpy as np
import sklearn
from random import shuffle

# This is the generator function that processes training or validation data,
# then yields the results btach by batch to the caller (fit_generator())
def generator(samples, batch_size=32, train=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #add center camera image and steering angle to output arrays
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # Apply data augmentation only for training data
                if train:
                    # add left & right camera images with a steering correction +/-0.2
                    name_l = './data/IMG/'+batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(name_l)
                    left_angle = center_angle+0.2
                    images.append(left_image)
                    angles.append(left_angle)
                    name_r = './data/IMG/'+batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(name_r)
                    right_angle = center_angle-0.2
                    images.append(right_image)
                    angles.append(right_angle)
                    # # augment data with flipped images
                    # for image, angle in zip([center_image, left_image, right_image], [center_angle, left_angle, right_angle]):
                    #     flipped_image = cv2.flip(image, 1)
                    #     images.append(image)
                    #     angles.append(-angle)

            # convert into numpy arrays
            X_train = np.array(images)
            y_train = np.array(angles)
            # return this batch to the caller
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32, train=True)
validation_generator = generator(validation_samples, batch_size=32, train=False)

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Here I'm using using the NVIDIA's DNN model for training,
# which consists of 5 convolution (with RELU activation) and 4 dense (linear) layers
# I also added Droput layers in-between to avoid or reduce overfitting

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
input_shape = (160, 320, 3)
# trim images to only see section with road -> output_shape = (80, 320, 3)
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=input_shape))
# normalize image data to the range [-1, 1]
model.add(Lambda(lambda x: x/127.5 - 1.))
# 3 conv2 layers with 5x5 filter size, 2x2 stride, and RELU activation
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.25)) #first droput
# 2 conv2 layers with 3x3 filter size and RELU activation
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.25)) #second dropout
model.add(Flatten())
# 4 dense layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5)) #third dropout
model.add(Dense(1))

# Compile the model using Adam optimizer
# And use MSE as a loss measure, which is suitable for regression data
model.compile(loss='mse', optimizer='adam')
# Train the model with fit_generator() instead of fit() for better speed perfrmoance and memory efficincy
history_object = model.fit_generator(train_generator, samples_per_epoch= \
            3*len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=5)

#save the model
model.save('model.h5')

"""
### print the keys contained in the history object
print(history_object.history.keys())

import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
"""
