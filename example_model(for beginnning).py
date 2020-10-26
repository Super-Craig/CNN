#!/usr/bin/env python
# coding: utf-8

# In[51]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
import pickle
import numpy as np

x = pickle.load(open("x_training.pickle","rb"))
y = pickle.load(open("y_training.pickle","rb"))

x_testing = pickle.load(open("x_testing.pickle","rb"))
y_testing = pickle.load(open("y_testing.pickle","rb"))

#x = np.array(x).reshape(-1,150,150,1)
x = x/255.0


# In[4]:


y


# In[ ]:





# In[16]:


print(len(x))


# In[56]:


model = Sequential()
model.add(Conv2D(64,(3,3), input_shape = (150,150,1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation("relu"))

#model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Activation("relu"))

#model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Activation("relu"))

model.add(Dense(3))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
             optimizer=SGD(lr=0.01, momentum=0.9),
             metrics=['accuracy'])


# In[25]:


y = np.array(y)
y


# In[93]:





# In[26]:


y.shape


# In[57]:


model.fit(x,y,batch_size=32,epochs=100, validation_data = (x_testing, y_testing))


# In[54]:


test_loss, test_acc = model.evaluate(x_testing,y_testing)
print("Tested Acc", test_acc)


# In[8]:


model.save('D:/keras_models')


# In[19]:


from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())


# In[8]:


import tensorflow as tf

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()


# In[ ]:




