import os, glob, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
# import keras.backend.tensorflow_backend as K

import tensorflow as tf

"""""""""
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)
"""""""""

x_train, x_test, y_train, y_test = np.load('./numpy_data/multi_image_data.npy',allow_pickle=True)

"""""""""
print(x_train.shape)
print(x_train.shape[0])
print(x_test.shape[0])
print(y_train.shape[0])
print(y_test.shape[0])
"""""""""

categories=[chr(ord("A")+i) for i in range(0,26)]
nb_classes = len(categories)

#Generation
x_train = x_train.astype(float) / 255
x_test = x_test.astype(float) / 255


model = Sequential()
model.add(Conv2D(32, (3, 3), padding = "same", input_shape = x_train.shape[1:], activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0,5))
model.add(Dense(nb_classes, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics =['accuracy'])
model_dir = './model'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_path = model_dir + '/multi_img_classification.model'
checkpoint = ModelCheckpoint(filepath = model_path, monitor = 'val_loss', verbose = 1, save_best_only = True)
early_stopping = EarlyStopping(monitor ='val_loss', patience = 6)

model.summary()

history = model.fit(x_train, y_train, batch_size = 32, epochs = 2,
                    validation_data = (x_test, y_test),
                    callbacks = [checkpoint, early_stopping])
print("정확도 : %.4f" % (model.evaluate(x_test, y_test)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker = '.', c = 'red', label = 'val_set_loss')
plt.plot(x_len, y_loss, marker = '.', c = 'blue', label = 'train_set_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()