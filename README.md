# Hiji Classifier

This repository holds our collected data, as well as different classifier systems.

## The Goal

We want to predict header impacts based on collected data

## Contents

### Data

- .csv files containing timestamp and acceleration/rate data
- filenames represent the date, followed by a .num if there were multiple files from the same day
- walking-only files are prepended with 'W-'
- impacts.txt contains timestamps for each recorded impact

### Preprocessing

- preprocessing.py takes in all data files and creates a final, preprocessed file that contains one second intervals, each with an impact flag
- extract_timestamps takes header videos, extract frames by milliseconds, and convert to isotimestamps

### Tools

- exploration provides some code to help determine how we need to filter our data during preprocessing
- lineplot_tools also helps with initial visualization
- train_test is where we store the list of important features using get_vars(), as well as general model tools to split data into train/test sets, evaluate the model, and plot confusion matrices

### Models

- ridge-regression detects which variables are most/least important for predicting head impact flag
- f-test performs linear regression
- model_comparision compares multiple models
- xgb, log_reg, and decision_gradient attempt to optimize and evaluate models
- kalman_filter uses Kalman filter and Mahalanobis distance to detect sudden movement changes