
# coding: utf-8

# In[55]:

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from skimage import io
from sklearn.decomposition import PCA


# In[56]:

# READING FILE

train = 'training_data/images_data.csv' # images_data = training_data
predict = 'training_data/unknown_data.csv' # unknown_data = prediction_data
training_data = pd.read_csv(train, index_col = 0)
prediction_data = pd.read_csv(predict, index_col = 0)

prediction_data = prediction_data.iloc[:100]


# In[57]:

# READING IMAGES

dfSize = training_data.shape[0]
images = io.imread_collection(training_data['filename'].values)
images = images.concatenate().reshape(dfSize, -1)

dfSize_p = prediction_data.shape[0]
images_p = io.imread_collection(prediction_data['filename'].values)
images_p = images_p.concatenate().reshape(dfSize_p, -1)


# In[58]:

# NORMALIZE DATA FUNCTION

def normalizeData(X):
    X = preprocessing.StandardScaler().fit_transform(X)
    return X


# In[59]:

# JOINING WEATHER + IMAGES
drop_columns = ['Weather', 'Date/Time', 'filename']
X_1 = training_data.drop(drop_columns, axis = 1) # 2244 x 7

# images = np.hsplit(images, 2)[1] # Splitting images decreases accuracy
X_2 = pd.DataFrame(images) # 2244 x 147456

X = pd.concat([X_1, X_2], axis = 1)
# X = PCA(50).fit_transform(X) # preprocess
y = training_data['Weather']

# Set up for prediction X
X_1 = prediction_data.drop(drop_columns, axis = 1)
X_2 = pd.DataFrame(images_p)
X_p = pd.concat([X_1, X_2], axis = 1)
# X_P = PCA(50).fit_transform(X_p) # preprocess


# In[60]:

# SVC Classifcation Model Code
components = 50
SVC_C = 0.01

SVC_model = make_pipeline(
    FunctionTransformer(normalizeData),
    PCA(n_components=components),
    SVC(kernel = 'linear', C = SVC_C),
)


# In[61]:

SVC_model.fit(X, y)


# In[62]:

y_p = SVC_model.predict(X_p)


# In[72]:

prediction_data['prediction'] = y_p
# print(prediction_data)

prediction_data.to_csv('prediction_data/prediction_data.csv')

