# Webcams, Predictions, and Weather

This project uses scikit-learn machine learning libraries to predict weather based off of images and weather features. The machine learning model uses existing data to train on to predict new weather off of images and weather features. This project may greatly benefit from using big data tools like PySpark to run its code on.

This project is made to work on the linux (Ubuntu) environment with 8gb of RAM. Please have at least 8gb of RAM to be able to run Training_model.py and Prediction_model.py.

Setup:
1. Run the following scripts in order (no arguments required): Training_data.py, filename_builder.py

Obtaining accuracy scores:
1. Run Training_model.py

Optional: Obtaining Kit Kam images sorted based on weather conditions of Clear, Cloudy Rain, Snow
1. Optional: Run Weather_separation.py to obtain a weather folder that categorizes known weather images.

*** DELETE ALL FILES in prediction_data folder to rerun ***
Optional: Bbtaining Prediction Images sorted by weather
1. Run Predicton_model.py to obtain a csv of the first 100 row weather prediction in prediction_data/prediction_data.csv
2. Run Prediction_seperation.py to obtain images sorted based on their weather in prediction_data folder

Training_model.py only outputs the accuracy scores for the test "SVM classification report w/ combined weather types". Obtaining the output for other tests requires manipuating some code in Training_data.py and re-running all the scripts. Getting tests for 'Only images' or 'Only weather features' just requires commenting and uncommenting a single line in Training_model.py