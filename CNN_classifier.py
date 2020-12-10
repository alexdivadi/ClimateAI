#!/usr/bin/env python
# coding: utf-8

# # **Background information:** 
# 
# Welcome to the coding challenging! This will be a 3-part challenge and you will develop an image processing and computer vision techniques to analyze the health of Solar photovoltaic (PV) systems. The challenege will be judged based on these  three results.
# 
# **(a)**Bench marking classification results that can be shared broadly with the community. 
# 
# 
# **(b)**Algorithm optimization towards real-time classification that can be used by low performance edge-computing devices.
# 
# ## **Bonus**
# **(c)** Incorporation of more classes and the ability to recognize anomalies that are outside of the 12 classes of InfraredSolarModules.
# 
# 
# We will be focusing on Solar photovoltaic (PV) datasets. Here's something we'd like you to know before you start:
# 
# We would like you to pre-process, visualize, perform an exploratory data analysis. Correlation analysis and anything you can think of are welcomed.
# 
# 
# 
# 
# If you have any questions, feel free to reach out to **@Tannistha** :)

# ## **Data downloads and requirements**:
# 
# We will use datasets from:
# https://github.com/RaptorMaps/InfraredSolarModules
# 
# 
# You are encouraged to use Python and its libraries for this challenge. For evaluation, we recommend numpy, pandas, matplotlib and seaborn for part a and Keras or PyTorch for part b. Please submit your work as one single .ipynb (recommended) or .py file.
# Please attach your model file that has weight (pkl) and a explain how to use it with new dataset.

# ### Link to previous work is available on 
# 
# *   InfraRed Thermography:
# 
# https://www.mdpi.com/1424-8220/20/4/1055
# 
# https://www.mdpi.com/1424-8220/20/4/1055
# *   Other relevant links
# 
# 
# 
# http://arxiv.org/abs/1807.02894
# 
# https://ai4earthscience.github.io/iclr-2020-workshop/papers/ai4earth22.pdf
# 
# https://onlinelibrary.wiley.com/doi/abs/10.1002/pip.3191
# 
# https://www.sciencedirect.com/science/article/abs/pii/S0038092X20308665
# 

# ### **Prepare Data** 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import json
import os
import random
import tensorflow as tf


# In[2]:


os.listdir('./')


# In[3]:


with open("./module_metadata.json", 'r') as read_file:
    data = json.load(read_file)


# In[4]:


df_files = pd.DataFrame(data)
df_files.head()


# In[5]:


def loadFiles(data):
    new_df = []
    for file in data.iloc[0]:
        img_data = img.imread(file)
        new_df.append(img_data / 255)
    return np.array(new_df)


# In[6]:


X = loadFiles(df_files)
X


# In[97]:


y = df_files.iloc[1]
y.to_numpy()


# In[10]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


# In[56]:


le = LabelEncoder()
le.fit(['No-Anomaly', 'Cell', 'Cell-Multi', 'Cracking', 'Diode', 'Diode-Multi', 'Hot-Spot', 'Hot-Spot-Multi', 'Offline-Module', 'Shadowing', 'Soiling', 'Vegetation'])
labels = le.transform(y)
classes = dict(enumerate(le.classes_))
print(classes)


# ### **Create Model** 

# In[98]:


img_rows, img_cols = 40, 24
image_shape = (40, 24, 1)
num_classes = 12


# In[99]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D

def classificationModel(filters_list, nodes_list, conv_filter_size, pool_filter_size, activation_list, dropout):

    model = Sequential()
    model.add(Conv2D(filters_list[0], conv_filter_size, activation=activation_list[0], padding='same', input_shape=image_shape))
    model.add(MaxPooling2D(pool_size=pool_filter_size, padding='same'))
    model.add(Conv2D(filters_list[1], conv_filter_size, activation=activation_list[0]))
    model.add(MaxPooling2D(pool_size=pool_filter_size))
  #  model.add(Conv2D(filters_list[2], conv_filter_size, activation=activation_list[0]))
  #  model.add(MaxPooling2D(pool_size=pool_filter_size))
    model.add(Flatten())
    model.add(Dense(nodes_list[0], activation=activation_list[0]))
    model.add(Dense(nodes_list[1], activation=activation_list[0]))
    model.add(Dropout(dropout))
    model.add(Dense(nodes_list[2], activation=activation_list[1]))
    
    return model
    


# In[100]:


model = classificationModel([20, 32, 48], [192, 48, num_classes], (1), (2), ['relu', 'softmax'], 0.25)
model.summary()


# In[101]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['AUC'])


# In[102]:


batch_size = 32
epochs = 8


# ### **Train Model**
# 
# Cross-validation done using the stratified K-fold method

# In[129]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


# In[130]:


skf = StratifiedKFold()
skf.get_n_splits(X,y)


# In[104]:


histories = []
predictions = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train.shape, y_train.shape)
    histories.append(model.fit(tf.expand_dims(X_train,-1), to_categorical(le.transform(y_train)), batch_size, epochs, 
                                validation_data=(tf.expand_dims(X_test,-1), to_categorical(le.transform(y_test)))))
    predictions.append(model.predict(tf.expand_dims(X_test,-1)))


# ### **Evaluate Results**

# In[110]:


def plotmetric(history, metric):
    plt.title("Model "+metric.title())
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric])
    plt.xlabel('epoch')
    plt.ylabel('metric')
    #plt.ylim([0,1])
    plt.show()


# In[111]:


for history in histories:
    plotmetric(history, 'auc')


# In[107]:


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report,multilabel_confusion_matrix


# In[149]:


def plot_roc_curve(y_true, y_pred, n):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(classes.get(n))
    plt.legend()
    plt.show()
    print(thresholds[np.argmax(tpr - fpr)])


# In[112]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[150]:


pred_proba = model.predict(tf.expand_dims(x_test,-1))
for n in classes.keys():
    plot_roc_curve(to_categorical(le.transform(y_test))[:,n], pred_proba[:,n], n)
    


# In[165]:


y_pred = (pred_proba > 0.5)
print(classification_report(to_categorical(le.transform(y_test)),y_pred, target_names=le.classes_))


# In[152]:


multilabel_confusion_matrix(to_categorical(le.transform(y_test)),y_pred)


# In[153]:


score = roc_auc_score(to_categorical(le.transform(y_test)), pred_proba, multi_class='ovo')
print(score)


# In[215]:


def predict_image_by_idx(i, threshold):
    holdout_image = X[i]
    prob = model.predict(holdout_image.reshape(1, 40, 24, 1))
    plt.imshow(holdout_image)
    pred = prob > threshold
    print(pred[0])
    n = le.classes_[pred[0]]
    plt.title("Predicted: " + ', '.join(n) + "\nActual: "+y[i])


# In[204]:


image_list = [(idx,value) for idx, value in enumerate(df_files.loc['anomaly_class']) if value=='Diode-Multi']
image = random.choice(image_list)
print(image)


# In[218]:


predict_image_by_idx(image[0], 0.1)


# In[ ]:




