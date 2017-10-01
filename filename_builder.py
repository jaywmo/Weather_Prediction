
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import re
import glob # For getting all filenames


# In[2]:

# Define re match string and open training_data and prediction_data
matchstring = r'\/katkam-(\d{4})(\d{2})(\d{2})(\d{2})'
weather = pd.read_csv('training_data/training_data.csv')
prediction = pd.read_csv('training_data/prediction_data.csv')


# In[3]:

# Grab filenames for all images
filenames = glob.glob("katkam-scaled/*.jpg") # For Linux


# In[4]:

# Create empty dataframe and insert filenames into dataframe
df = pd.DataFrame(0, index=np.arange(len(filenames)), columns=['filename', 'Year', 'Month', 'Day', 'Time'])
df['filename'] = filenames


# In[5]:

# Find the day, month, time, and year form Date/Time to be used for inner join of filenames and data
def findDate(x):
    match = re.search(matchstring, x[0])
    x[1] = (int)(match.group(1))
    x[2] = (int)(match.group(2))
    x[3] = (int)(match.group(3))
    x[4] = (int)(match.group(4))
    return x


# In[6]:

# Apply findDate function
df = df.apply(findDate, axis=1)


# In[7]:

training = weather.merge(df, left_on=['Day', 'Month', 'Time', 'Year'], right_on=['Day', 'Month', 'Time', 'Year'], how='inner').reset_index()
unknown = prediction.merge(df, left_on=['Day', 'Month', 'Time', 'Year'], right_on=['Day', 'Month', 'Time', 'Year'], how='inner').reset_index()


# In[8]:

# Cleaning Training Data
del training['Unnamed: 0']
del training['index']
# Cleaning Prediction Data
del unknown['Unnamed: 0']
del unknown['index']

training.to_csv('training_data/images_data.csv')
unknown.to_csv('training_data/unknown_data.csv')


# In[ ]:



