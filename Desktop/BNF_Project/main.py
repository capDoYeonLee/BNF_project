from types import new_class
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from utils import get_folded_weights_1st, get_folded_weights_2nd, get_folded_weights_3rd
from utils import get_folded_weights_4th, get_folded_weights_5th, get_folded_weights_6th
from model import Batch_Normalization_Folding_Model
from make_weight_model import make_model_for_wb
from dataset import data_loader
import os, datetime

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

x_train, x_test, y_train, y_test = data_loader()
i, x = Batch_Normalization_Folding_Model(x_train, y_train)


real_model = Model(i, x)
real_model.summary()


# Compile
real_model.compile(optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy',])

train_start = time.time()


# Fit
r = real_model.fit(
x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32, callbacks=tensorboard_callback)  


train_end = time.time()

round_time = round( (train_end - train_start), 5)
print("training time : {} second".format( round_time ))


plt.plot(r.history['accuracy'], label='acc', color='red')
plt.plot(r.history['val_accuracy'], label='val_acc', color='green')
plt.legend()
plt.show()



# label mapping

labels = '''airplane automobile bird cat deerdog frog horseship truck'''.split()

# select the image from our test dataset
image_number = 0

# display the image
plt.imshow(x_test[image_number])

# load the image in an array
n = np.array(x_test[image_number])

# reshape it
p = n.reshape(1, 32, 32, 3)

# pass in the network for prediction and
# save the predicted label
predicted_label = labels[real_model.predict(p).argmax()]

# load the original label
original_label = labels[y_test[image_number]]

# display the result
print("Original label is '{}' \n predicted label is '{}' ".format(
	original_label, predicted_label))
