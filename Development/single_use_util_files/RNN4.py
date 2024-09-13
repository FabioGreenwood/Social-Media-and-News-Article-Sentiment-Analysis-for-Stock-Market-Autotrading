import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import timeseries_dataset_from_array
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import keras
import numpy as np
import pandas as pd

import copy




# Example multivariate time series data (replace this with your actual time series)
# Assuming two features in the time series


# Create random multivariate time series data
df_X = pd.read_csv(r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\precalculated_assets\example_input.csv")
df_X.set_index(df_X.columns[0], inplace=True)
df_y = pd.read_csv(r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\precalculated_assets\example_output.csv")
df_y.set_index(df_y.columns[0], inplace=True)

keras.utils.timeseries_dataset_from_array


sequence_length = 10
input_shape = (sequence_length, df_X.shape[1])
batch_size = 32

time_series_dataset = timeseries_dataset_from_array(
        df_X.values,
        df_y.values,
        sequence_length=sequence_length,
        sampling_rate=1,
        batch_size=batch_size,
        shuffle=True
    )



model = Sequential([
    LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape),
    LSTM(units=50, activation='relu', return_sequences=True),
    LSTM(units=50, activation='relu', return_sequences=True),
    Dense(units=1)
])
## Compile the model
#model.compile(optimizer='adam', loss='mse')
#
## Define the model
#model = Sequential([
#    LSTM(units=50, activation='relu', input_shape=input_shape),
#    Dense(units=1)
#])

# Compile the model
model.compile(optimizer='adam', loss='mae')

# Train the model using the time series dataset
model.fit(time_series_dataset, epochs=2)


# "validate" the model
validation_dataset = timeseries_dataset_from_array(df_X.values, df_y.values, sequence_length=1, sampling_rate=1, shuffle=False)
validation_score = model.evaluate(validation_dataset)
print("Validation Loss:", validation_score)

# "test" the model
blank_y = copy.deepcopy(df_y)
blank_y.iloc[:,0] = 0
test_dataset = timeseries_dataset_from_array(df_X, blank_y, sequence_length=2, sampling_rate=1, shuffle=False)
predictions = model.predict(test_dataset)
#predictions = model.predict(df_X)
#mae_score = mean_squared_error(df_y, predictions)
#print("mae_score:", validation_score)

print("hello")