
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import glob


# In[2]:

input_directory = 'yvr-weather'
training_output_file = 'training_data/training_data.csv'
prediction_output_file = 'training_data/prediction_data.csv'
csv_files = glob.glob(input_directory + "/*.csv") # Reads every .csv file in input_directory
training_csv_list = [] # List used to concat the dataframes
prediction_csv_list = []
# TODO: Fix code to not have dataframes to append to list

for csv in csv_files:
    data = pd.read_csv(csv, header = 14) 
    data = data.drop(['Data Quality'], 1) # Drop column 'Data Quality'
    data = data.dropna(axis = 1, how = 'all') # Drop column if every value is null
    
    prediction = data.loc[data['Weather'].isnull()]  
    prediction_csv_list.append(prediction)
    
    data = data.loc[~data['Weather'].isnull()] # Drop row if weather is null
    training_csv_list.append(data)


# In[3]:

# Creates training data
training_data = pd.concat(training_csv_list, ignore_index = True)

# Creates prediction data
prediction_data = pd.concat(prediction_csv_list, ignore_index = True)


# In[4]:

# Training data cleaning

# Parsing Time
training_data['Time'] = training_data['Time'].str.slice(start = 0, stop = 2)
training_data['Time'] = training_data['Time'].astype(int)
training_data = training_data[ (training_data['Time'] <= 21) & (training_data['Time'] >= 6)]
training_data = training_data.reset_index(drop = True)

# Removing NaN data
training_data = training_data.drop(['Hmdx', 'Wind Chill'], 1)
training_data = training_data.dropna(axis=0, how='any')

# Prediction data cleaning
prediction_data['Time'] = prediction_data['Time'].str.slice(start = 0, stop = 2)
prediction_data['Time'] = prediction_data['Time'].astype(int)
prediction_data = prediction_data[ (prediction_data['Time'] <= 21) & (prediction_data['Time'] >= 6)]
prediction_data = prediction_data.reset_index(drop = True)
prediction_data.loc[:, 'Weather'] = 0 # We don't want NULL values in weather
prediction_data = prediction_data.drop(['Hmdx', 'Wind Chill'], 1)
prediction_data = prediction_data.dropna(axis=0, how='any')


# In[5]:

# REMOVE RESULTS WITH NOT ENOUGH SAMPLE PHOTOS

training_data = training_data.groupby('Weather').filter(lambda x: len(x) >= 20)
training_data = training_data.reset_index(drop = True)


# In[6]:

# COMBINE Clear with Mainly Clear
# COMBINE Rain with Moderate Rain
# COMBINE Rain with Rain Showers + Rain,Fog

training_data['Weather'] = training_data['Weather'].replace('Mainly Clear', 'Clear')
training_data['Weather'] = training_data['Weather'].replace('Moderate Rain', 'Rain')
training_data['Weather'] = training_data['Weather'].replace('Rain Showers', 'Rain')
training_data['Weather'] = training_data['Weather'].replace('Rain,Fog', 'Rain')
training_data['Weather'] = training_data['Weather'].replace('Mostly Cloudy', 'Cloudy')


# In[7]:

# SAVE

training_data.to_csv(training_output_file)

prediction_data.to_csv(prediction_output_file)


# In[ ]:



