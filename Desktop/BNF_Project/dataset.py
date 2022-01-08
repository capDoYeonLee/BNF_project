from types import new_class
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
import ssl 



def data_loader():
    
  ssl._create_default_https_context = ssl._create_unverified_context

  cifar10 = tf.keras.datasets.cifar10
 
  # Distribute it to train and test set
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


  # Reduce pixel values
  x_train, x_test = x_train / 255.0, x_test / 255.0
  # flatten the label values
  y_train, y_test = y_train.flatten(), y_test.flatten()
  
  return x_train, x_test, y_train, y_test
