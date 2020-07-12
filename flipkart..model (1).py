#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
import keras 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.keras.preprocessing import image


# In[7]:


img = image.load_img("D:/dataset/flipkart/train/jeans/jeans (72).jpeg")
plt.imshow(img)


# In[8]:


cv2.imread("D:/dataset/flipkart/train/jeans/jeans (72).jpeg").shape


# In[9]:


cv2.imread("D:/dataset/flipkart/train/jeans/jeans (72).jpeg")


# In[10]:


train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)


# In[11]:


train_dataset = train.flow_from_directory("D:/dataset/flipkart/train",
                                     target_size = (200,200),
                                     batch_size = 10,
                                     class_mode = 'binary')
validation_dataset = train.flow_from_directory("D:/dataset/flipkart/validation",
                                     target_size = (200,200),
                                     batch_size = 10,
                                     class_mode = 'binary')
                                     


# In[12]:


train_dataset.class_indices


# In[13]:


train_dataset.classes


# In[14]:


from tensorflow.keras import optimizers


# In[15]:


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3), activation = 'relu',input_shape = (200,200,3)),
                                                           tf.keras.layers.MaxPool2D(2,2),
                                                           tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                                            tf.keras.layers.MaxPool2D(2,2),
                                                            tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                                            tf.keras.layers.MaxPool2D(2,2),
                                                            tf.keras.layers.Flatten(),
                                                            tf.keras.layers.Dense(128,activation = 'relu'),
                                                            tf.keras.layers.Dense(1,activation = 'sigmoid')])
                                                            
                                                            


# In[16]:


model.compile(loss= 'binary_crossentropy',
            optimizer= tf.keras.optimizers.RMSprop(lr=0.001),
             metrics =['accuracy'])


# In[17]:


print(model.summary())


# In[18]:


from keras import backend as k


# In[19]:


model_fit = model.fit(train_dataset, steps_per_epoch = 100, epochs =15 , validation_data = validation_dataset)


# In[20]:


plt.plot(model_fit.history['acc'])
plt.plot(model_fit.history['val_acc'])
plt.title('model acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','validation'], loc='upper left')
plt.show()


# In[22]:


dir_path = 'D:/dataset/flipkart/test'
for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+ i, target_size = (200,200))
    plt.imshow(img)
    plt.show()
    x = image.img_to_array(img)
    x =np.expand_dims(x,axis = 0)
    images = np.vstack([x])
    pred = model.predict(images)
    if pred == 0:
        print('its a jeans')
    else:
        print('its a trouser')

    


# In[ ]:




