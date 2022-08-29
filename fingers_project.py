import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def extract_label(base_path):
    """

    Parameters
    ----------
    base_path

    Returns
    -------

    """
    path = []
    label = []
    for filename in os.listdir(base_path):
        label.append(filename.split('.')[0][-2])
        path.append(base_path + filename)
    return path, label


def data_sets_split(catalog):
    """

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


train_base = 'C:/Users/jkche/PycharmProjects/project_fingers/train/'
test_base = 'C:/Users/jkche/PycharmProjects/project_fingers/test/'

train_set_path, train_set_label = extract_label(train_base)
test_set_path, test_set_label = extract_label(test_base)

print(f'Training set: {len(train_set_label)}')
print(f'Test set: {len(test_set_label)}')

# Pokazanie przykładowego zdjęcia, jego labelki oraz rozmiaru
photo_nr = 103
image = cv2.imread(train_set_path[photo_nr])
image_label = train_set_label[photo_nr]

print(f'wymiary obrazka: {image.shape}')
print(f'liczba palców na dłoni to: {image_label}')
print(image.shape)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# tworzenie zestawu treningowego i testowego
print('Tworzę listę treningową i testową...')
X_train = data_sets_split(train_set_path)
X_test = data_sets_split(test_set_path)

# przerobienie listy trenigowej i testowej na tablice np
X_train = np.array(X_train)
X_test = np.array(X_test)

print(f'X_train shape = {X_train.shape}')
print(f'X_test shape = {X_test.shape}')
