# The following code separates known weather type images to separate folders. This served as a way to look at a high level how images not taken in weather station might affect classified weather.

import pandas as pd
import numpy as np
from shutil import copyfile
import os

# READING FILE
input_file = 'prediction_data/prediction_data.csv'
testing_data = pd.read_csv(input_file, index_col = 0)

# Creating folders
for weather in testing_data['prediction'].unique():
    os.makedirs('prediction_data/' + weather)

# Copying images to folders
for weather in testing_data['prediction'].unique():
    weather_df = testing_data[testing_data['prediction'] == weather]
    image_list = weather_df['filename'].values
    destination_folder = 'prediction_data/' + weather

    for image in image_list:
        src = image
        dst = destination_folder + image[13:]
        copyfile(src, dst)
