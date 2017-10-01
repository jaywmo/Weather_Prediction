
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
# import sys
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from skimage import io
from sklearn.decomposition import PCA


# In[2]:

# READING FILE

input_file = 'training_data/images_data.csv' # images_data = training_data
testing_data = pd.read_csv(input_file, index_col = 0)


# In[3]:

# READING IMAGES

dfSize = testing_data.shape[0]
images = io.imread_collection(testing_data['filename'].values)
images = images.concatenate().reshape(dfSize, -1)


# In[4]:

# NORMALIZE DATA FUNCTION

def normalizeData(X):
    X = preprocessing.StandardScaler().fit_transform(X)
    return X


# In[5]:

# JOINING WEATHER + IMAGES
drop_columns = ['Weather', 'Date/Time', 'filename']
X_1 = testing_data.drop(drop_columns, axis = 1) # 2244 x 7

# images = np.hsplit(images, 2)[1] # Splitting images decreases accuracy
X_2 = pd.DataFrame(images) # 2244 x 147456

X = pd.concat([X_1, X_2], axis = 1)

# TRAINING SPLIT

y = testing_data['Weather']
X_train, X_test, y_train, y_test = train_test_split(X,y)
# X_train, X_test, y_train, y_test = train_test_split(X_1,y) # Only weather data
# X_train, X_test, y_train, y_test = train_test_split(X_2,y) # Only images


# In[6]:

# print(X)


# In[7]:

# TRAINING MODELS

components = 50

# GaussianNB
NB_model = make_pipeline(
    FunctionTransformer(normalizeData),
    PCA(n_components=components),
    GaussianNB(),
)

# KNeighborsClassifier
KN_N = 20
KN_model = make_pipeline(
    FunctionTransformer(normalizeData),
    PCA(n_components=components),
    KNeighborsClassifier(n_neighbors = KN_N),
)

# SVC
SVC_C = 0.01
SVC_model = make_pipeline(
    FunctionTransformer(normalizeData),
    PCA(n_components=components),
    SVC(kernel = 'linear', C = SVC_C),
)


# In[8]:

# ACCURACY SCORE

# NB_model.fit(X_train, y_train)
# print ('GaussianNB: %f' % (NB_model.score(X_test, y_test)) )
# print(classification_report(y_test, NB_model.predict(X_test)))


# In[9]:

# KN_model.fit(X_train, y_train)
# print ('KNeighborsClassifier: %f' % (KN_model.score(X_test, y_test)) )
# print(classification_report(y_test, KN_model.predict(X_test)))


# In[10]:

SVC_model.fit(X_train, y_train)
print ('SVC: %f' % (SVC_model.score(X_test, y_test)) )
print(classification_report(y_test, SVC_model.predict(X_test)))


# In[ ]:



