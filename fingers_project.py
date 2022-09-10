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


def extract_label(base_path):
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


train_set_path, _, _,  train_set_label = extract_label(TRAIN_DIR_PATH)
test_set_path, _, _, test_set_label = extract_label(TEST_DIR_PATH)

print(f'Training set: {len(train_set_label)}')
print(f'Test set: {len(test_set_label)}')

# Pokazanie przykładowego zdjęcia, jego labelki oraz rozmiaru
photo_nr = 103
image = cv2.imread(train_set_path[photo_nr])
image_label = train_set_label[photo_nr]

print(f'Image shape: {image.shape}')
print(f'Numbers of fingers: {image_label}')
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

print(f'Encoded train labels set unique values: {np.unique(y_train)}')
print(f'Encoded test labels set unique values: {np.unique(y_test)}')
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
