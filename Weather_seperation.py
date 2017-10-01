
# coding: utf-8

# In[99]:

# The following code separates known weather type images to separate folders. This served as a way to look at a high level how images not taken in weather station might affect classified weather.

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import sys
# import re
from shutil import copyfile
import os


# In[100]:

# READING FILE

input_file = 'training_data/images_data.csv' # images_data = training_data
testing_data = pd.read_csv(input_file, index_col = 0)


# In[101]:

# CHECKING FOR WEATHER DEFINITION

print(testing_data['Weather'].unique())


# In[102]:

# CHECKING FOR WEATHER DEFINITION

for weather in testing_data['Weather'].unique():
    os.makedirs('weather/' + weather)

for weather in testing_data['Weather'].unique():
    weather_df = testing_data[testing_data['Weather'] == weather]
    image_list = weather_df['filename'].values
    destination_folder = 'weather/' + weather
    
    for image in image_list:
        src = image
        dst = destination_folder + image[13:]
        copyfile(src, dst)

# cloudy = testing_data[testing_data['Weather'] == 'Cloudy'] # 469
# mostly_cloud = testing_data[testing_data['Weather'] == 'Mostly Cloudy'] #425

# print(cloudy['filename'])


# In[103]:

# list = cloudy['filename'].values
# destination_folder = 'weather/cloudy'
# for image in list:
#     src = image
#     dst = destination_folder + image[13:]
#     copyfile(src, dst)

