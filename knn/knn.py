# Ronggang Cui
# 1617665
# Sep 15, 2022
# Assignment 1, CMPUT 328

import torch
from torchvision import datasets, transforms
import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat
from torch import linalg as LA

def knn(x_train, y_train, x_test, n_classes, device):
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 1000 x 784 testing images
    n_classes: no. of classes in the classification task
    device: pytorch device on which to run the code
    return: predicted y_test which is a 1000-sized vector
    """
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 5000 x 784 testing images
    return: predicted y_test which is a 5000 vector
    """

    # get the shape size
    num_row_train, num_col_train = x_train.shape
    num_row_test, num_col_test = x_test.shape

    # initialize the output y_test
    y_test = np.zeros(num_row_test)

    # convert to float type
    x_test = x_test.astype(np.float32)
    x_train = x_train.astype(np.float32)

    # Convert from numpy to pytorch tensor
    # Move the data to gpu
    x_test_tt = torch.from_numpy(x_test).to(device)
    x_train_tt = torch.from_numpy(x_train).to(device)
    distance = torch.zeros(num_row_train).to(device)

    # For every test data point, do classification
    for i in list(range(0, num_row_test)):
      
      distance_vector = x_train_tt - x_test_tt[i].repeat(num_row_train, 1)
      distance = LA.norm(distance_vector, dim=1)
      
      dist, distIndex = torch.topk(distance, 3, largest=False)
      
      closestY = torch.zeros(3)

      # get the labels of closest data points
      for k in list(range(0, 3)):
        closestY[k] = y_train[distIndex[k]]

      # use the mode label
      mode, modeIndex = torch.mode(closestY)
      y_test[i] = mode

    return y_test
