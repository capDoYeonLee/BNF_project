from types import new_class
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from utils import get_folded_weights_1st, get_folded_weights_2nd, get_folded_weights_3rd
from utils import get_folded_weights_4th, get_folded_weights_5th, get_folded_weights_6th
from make_weight_model import make_model_for_wb
from dataset import data_loader

x_train, x_test, y_train, y_test = data_loader()

model = make_model_for_wb(x_train)

w_1, b_1 = get_folded_weights_1st(model.layers[1], model.layers[2])
w_2, b_2 = get_folded_weights_2nd(model.layers[3], model.layers[4])
w_3, b_3 = get_folded_weights_3rd(model.layers[6], model.layers[7])
w_4, b_4 = get_folded_weights_4th(model.layers[8], model.layers[9])
w_5, b_5 = get_folded_weights_5th(model.layers[11], model.layers[12])
w_6, b_6 = get_folded_weights_6th(model.layers[13], model.layers[14])




def Batch_Normalization_Folding_Model(x_train, y_train):
    # Batch Normalization Folding Model
    K = len(set(y_train))
    i = Input(shape=x_train[0].shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', weights=[w_1,b_1],)(i)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', weights=[w_2, b_2])(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', weights=[w_3,b_3])(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', weights=[w_4,b_4])(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', weights=[w_5,b_5])(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', weights=[w_6,b_6])(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)

    # Hidden layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)

    # last hidden layer i.e.. output layer
    x = Dense(K, activation='softmax')(x)
    
    return i, x