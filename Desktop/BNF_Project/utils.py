from types import new_class
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from make_weight_model import make_model_for_wb
from dataset import data_loader

x_train, x_test, y_train, y_test = data_loader()
model = make_model_for_wb(x_train)




def get_folded_weights_1st(conv : model.layers[1], bn : model.layers[2]):
  epsilon = 1e-3
  gamma     =  bn.gamma
  beta      =  bn.beta
  mean      =  bn.moving_mean
  variance  =  bn.moving_variance
  kernel   =  conv.kernel
  bias     =  conv.bias
  new_weights = ( kernel * gamma / tf.sqrt(variance + epsilon) )
  new_bias    = beta + (bias - mean) * gamma / tf.sqrt(variance + epsilon)

  return new_weights, new_bias
  
def get_folded_weights_2nd(conv : model.layers[3], bn : model.layers[4]):
  epsilon = 1e-3
  gamma     =  bn.gamma
  beta      =  bn.beta
  mean      =  bn.moving_mean
  variance  =  bn.moving_variance
  kernel   =  conv.kernel
  bias     =  conv.bias
  new_weights = ( kernel * gamma / tf.sqrt(variance + epsilon) )
  new_bias    = beta + (bias - mean) * gamma / tf.sqrt(variance + epsilon)

  return new_weights, new_bias

def get_folded_weights_3rd(conv : model.layers[6], bn : model.layers[7]):
  epsilon = 1e-3
  gamma     =  bn.gamma
  beta      =  bn.beta
  mean      =  bn.moving_mean
  variance  =  bn.moving_variance
  kernel   =  conv.kernel
  bias     =  conv.bias
  new_weights = ( kernel * gamma / tf.sqrt(variance + epsilon) )
  new_bias    = beta + (bias - mean) * gamma / tf.sqrt(variance + epsilon)

  return new_weights, new_bias

def get_folded_weights_4th(conv : model.layers[8], bn : model.layers[9]):
  epsilon = 1e-3
  gamma     =  bn.gamma
  beta      =  bn.beta
  mean      =  bn.moving_mean
  variance  =  bn.moving_variance
  kernel   =  conv.kernel
  bias     =  conv.bias
  new_weights = ( kernel * gamma / tf.sqrt(variance + epsilon) )
  new_bias    = beta + (bias - mean) * gamma / tf.sqrt(variance + epsilon)

  return new_weights, new_bias

def get_folded_weights_5th(conv : model.layers[11], bn : model.layers[12]):
  epsilon = 1e-3
  gamma     =  bn.gamma
  beta      =  bn.beta
  mean      =  bn.moving_mean
  variance  =  bn.moving_variance
  kernel   =  conv.kernel
  bias     =  conv.bias
  new_weights = ( kernel * gamma / tf.sqrt(variance + epsilon) )
  new_bias    = beta + (bias - mean) * gamma / tf.sqrt(variance + epsilon)

  return new_weights, new_bias


def get_folded_weights_6th(conv : model.layers[13], bn : model.layers[14]):
  epsilon = 1e-3
  gamma     =  bn.gamma
  beta      =  bn.beta
  mean      =  bn.moving_mean
  variance  =  bn.moving_variance
  kernel   =  conv.kernel
  bias     =  conv.bias
  new_weights = ( kernel * gamma / tf.sqrt(variance + epsilon) )
  new_bias    = beta + (bias - mean) * gamma / tf.sqrt(variance + epsilon)

  return new_weights, new_bias 
