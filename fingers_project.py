import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers as tfl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

TRAIN_DIR_PATH = 'C:/Users/jkche/PycharmProjects/project_fingers/train/'
TEST_DIR_PATH = 'C:/Users/jkche/PycharmProjects/project_fingers/test/'


def extract_img_info(base_path):
    """
    Function to extract labels and path form names of images
    Parameters
    ----------
    base_path

    Returns
    -------

    """
    path = []
    fingers = []
    hand = []
    hand_fingers = []
    for filename in os.listdir(base_path):
        fingers.append(filename.split('.')[0][-2:-1])
        hand.append(filename.split('.')[0][-1])
        hand_fingers.append(filename.split('.')[0][-2:])
        path.append(base_path + filename)
    return path, fingers, hand, hand_fingers


def data_sets_split(catalog):
    """
    Function to split train and test sets
    Parameters
    ----------
    catalog

    Returns
    -------

    """
    data_set = []
    for path in catalog:
        img = cv2.imread(path)
        data_set.append(img)
    return data_set


train_set_path, _, _,  train_set_label = extract_img_info(TRAIN_DIR_PATH)
test_set_path, _, _, test_set_label = extract_img_info(TEST_DIR_PATH)

print(f'Training set: {len(train_set_label)}')
print(f'Test set: {len(test_set_label)}')

# Showing one of images, description and size
photo_nr = 103
image = cv2.imread(train_set_path[photo_nr])
image_label = train_set_label[photo_nr]

print(f'Image shape: {image.shape}')
print(f'Fingers and hand: {image_label}')
print(image.shape)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('Making train and test sets...')
X_train = data_sets_split(train_set_path)
X_test = data_sets_split(test_set_path)

X_train = np.array(X_train)
X_test = np.array(X_test)

print(f'X_train shape = {X_train.shape}')
print(f'X_test shape = {X_test.shape}')

label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(train_set_label)
y_test = label_encoder.transform(test_set_label)

print(f'Encoded train labels set unique values:\t {np.unique(y_train)}')
print(f'Encoded test labels set unique values:\t {np.unique(y_test)}')
if len(np.unique(y_train)) == len(np.unique(y_test)):
    categorical_number = len(np.unique(y_test))
elif len(np.unique(y_train)) > len(np.unique(y_test)):
    categorical_number = len(np.unique(y_train))
elif len(np.unique(y_train)) < len(np.unique(y_test)):
    categorical_number = len(np.unique(y_test))

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.25, random_state=22)
print(f'Values of images in train set for each category:\n{pd.Series(y_train).value_counts()}')
print(f'Values of images in validation set for each category:\n{pd.Series(y_val).value_counts()}')
print(f'Values of images in test set for each category:\n{pd.Series(y_test).value_counts()}')

# One hot encoding y labels for each sets
y_train = keras.utils.to_categorical(y_train, categorical_number)
y_val = keras.utils.to_categorical(y_val, categorical_number)
y_test = keras.utils.to_categorical(y_test, categorical_number)


# Convolution model
def convolutional_model(input_shape):
    input_img = tf.keras.Input(shape=input_shape)

    conv_layer1 = tfl.Conv2D(filters=8, kernel_size=4, strides=1, padding='same')(input_img)
    activation1 = tfl.ReLU()(conv_layer1)
    pool_layer1 = tfl.MaxPool2D(pool_size=8, strides=8, padding='same')(activation1)

    conv_layer2 = tfl.Conv2D(filters=16, kernel_size=2, strides=1, padding='same')(pool_layer1)
    activation2 = tfl.ReLU()(conv_layer2)
    pool_layer2 = tfl.MaxPool2D(pool_size=4, strides=4, padding='same')(activation2)

    conv_layer3 = tfl.Conv2D(filters=32, kernel_size=2, strides=1, padding='same')(pool_layer2)
    activation3 = tfl.ReLU()(conv_layer3)
    pool_layer3 = tfl.MaxPool2D(pool_size=4, strides=4, padding='same')(activation3)

    flatten = tfl.Flatten()(pool_layer3)
    outputs = tfl.Dense(units=12, activation='softmax')(flatten)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


conv_model = convolutional_model((128, 128, 3))
conv_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
print(conv_model.summary())

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(90)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(90)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(90)

history = conv_model.fit(train_dataset, epochs=5, validation_data=val_dataset)

conv_model.evaluate(X_test, y_test)
