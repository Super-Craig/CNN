# -*- coding: utf-8 -*-
"""Copy of Animal_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_i2m8ShsR1k2YRnxKrCIAdHPLUQjAqW4
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My\ Drive/Colab\ Notebooks

!pwd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
import pickle
import numpy as np
from tensorflow import keras

x = pickle.load(open("x_training.pickle","rb"))
y = pickle.load(open("y_training.pickle","rb"))

x_testing = pickle.load(open("x_testing.pickle","rb"))
y_testing = pickle.load(open("y_testing.pickle","rb"))

#x = np.array(x).reshape(-1,150,150,1)
x = x/255.0

from keras.applications.vgg16 import VGG16
from keras.models import Model

from tensorflow.keras.optimizers import Adam

print(len(x))

#model = Sequential()

model = VGG16(include_top=False, input_shape=(150,150,3))
#mark loaded layers as not trainable
for layer in model.layers:
	layer.trainable = False
#model.add(Conv2D(256,(3,3), input_shape = (150,150,1)))





#model.add(Conv2D(64,(3,3),input_shape=x.shape[1:]))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(64,(3,3)))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(32,(3,3)))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(16,(3,3)))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(8,(3,3)))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))





# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(64, activation='relu')(flat1)
class2 = Dense(32, activation='relu')(class1)
class3 = Dense(16, activation='relu')(class2)
output = Dense(3, activation='softmax')(class3)
 #define new model
model = Model(inputs=model.inputs, outputs=output)





#model.add(Flatten())
#model.add(Dropout(0.2))
#model.add(Dense(64))
#model.add(Activation("relu"))

#model.add(Dropout(0.2))
#model.add(Dense(64))
#model.add(Activation("relu"))

#model.add(Dropout(0.2))
#model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
#model.add(Activation("relu"))

#model.add(Dense(3))
#model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy",
            optimizer=SGD(learning_rate=0.01),
             metrics=['accuracy'])

model.summary()

y = np.array(y)
y



y.shape

model.fit(x,y,batch_size=32,epochs=50, validation_data = (x_testing, y_testing))

model.save("model_test")

"""# New Section

# New Section
"""

test_loss, test_acc = model.evaluate(x_testing,y_testing)
print("Tested Acc", test_acc)

model.save('D:/keras_models')

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())

import tensorflow as tf

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

