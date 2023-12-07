import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import timeseries_dataset_from_array
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import keras
import numpy as np
import pandas as pd
import matplotlib as plt
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime

# parameters
master_folder_path = r"C:\Users\Public\fabio_uni_work\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading"
example_input_path = os.path.join(master_folder_path, "example_input.csv")
date_format = '%d/%m/%Y %H:%M'

list_of_precalculated_asset_folders = ["always_up_down_results", "annotated_tweets", "cleaned_tweets_ready_for_subject_discovery", "experimental_records", "predictive_model", "sentiment_data", "technical_indicators", "temp_testing_dicts", "topic_models"]

# Example multivariate time series data (replace this with your actual time series)
# Assuming two features in the time series

# Create random multivariate time series data
df_X = pd.read_csv(example_input_path)
df_X.set_index(df_X.columns[0], inplace=True)
#df_y = pd.read_csv(r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\precalculated_assets\example_output.csv")
#df_y.set_index(df_y.columns[0], inplace=True)
pred_steps = 4

df_X= df_X[['£_open', '£_high', '£_low', '£_close', '£_volume', '$_sma_5', '$_sma_144', '$_ema_144', '~senti_score_t1', '~senti_score_t2', '~senti_score_t3', '~senti_score_t4', '~senti_score_t5']]
new_length = int(len(df_X) * 0.2)
print("new length: {}".format(new_length))
df_X = df_X[:new_length]

df_y = df_X["£_close"].shift(-pred_steps)
df_X = df_X[:-pred_steps]
df_y = df_y[:-pred_steps]

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

df_X_normalized = scaler_X.fit_transform(df_X)
df_y_normalized = scaler_y.fit_transform(df_y.values.reshape(-1, 1))


sequence_length = 10
input_shape = (sequence_length, df_X.shape[1])
batch_size = 32


training_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        df_X_normalized,
        df_y_normalized,
        sequence_length,
        batch_size=1,
        shuffle=False
    )

datetime_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        df_X.index,
        df_y_normalized,
        sequence_length,
        batch_size=1,
        shuffle=False
    )


## remove sequences that straddle a day
#for id in range(len(training_generator)):
#    if not datetime_generator[id][0][0][0][:10] == datetime_generator[id][0][0][-1][:10]:
#        training_generator[id] = None, None


def return_filtered_batches_that_dont_cross_two_days(training_generator, datetime_generator):
    mask = []
    for batch_x, output in datetime_generator:
        if datetime.strptime(batch_x[0][0], date_format).day == datetime.strptime(batch_x[0][-1], date_format).day:
            mask += [True]
        else:
            mask += [False]
    filtered_training_generator_batch_list = []
    for training_batch, Bool in zip(training_generator, mask):
        if Bool == True:
            filtered_training_generator_batch_list += [training_batch]
    return filtered_training_generator_batch_list

import random

def return_filtered_batches_that_dont_cross_two_days_v2(training_generator, datetime_generator):
    mask, new_data, new_targets = [], np.empty((0,training_generator.data.shape[1])), np.empty((0,training_generator.targets.shape[1]))
    for batch_x, output in datetime_generator:
        if datetime.strptime(batch_x[0][0], date_format).day == datetime.strptime(batch_x[0][-1], date_format).day:
            mask += [True]
        else:
            mask += [False]
    for data_n, target_n, Bool_n in zip(training_generator.data, training_generator.targets, mask):
        if Bool_n == True:
            new_data    = np.append(new_data, [data_n], axis=0)
            new_targets = np.append(new_targets, [target_n], axis=0)
    print(str(sum(mask)) + str(len(mask)))
    # replace removed batches with random batches
    for i in range(sum(mask), len(mask)):
        random_index = random.randint(0, sum(mask))
        new_data    = np.append(new_data, [new_data[random_index]], axis=0)
        new_targets = np.append(new_targets, [new_targets[random_index]], axis=0)
    #the time series generator's final batches tend to be blank, causing the first for loop to skip them, it is best to just transfer these directly
    for i in range(len(new_data), training_generator.data.shape[0]):
        new_data    = np.append(new_data, [training_generator.data[i]], axis=0)
        new_targets = np.append(new_targets, [training_generator.targets[i]], axis=0)

    training_generator.data     = new_data
    training_generator.targets  = new_targets
    return training_generator



## Filter batches that cross two days
#filtered_batches = [batch for batch in generator if not batch_crosses_two_days(batch)]

filtered_training_generator_batch_list = return_filtered_batches_that_dont_cross_two_days_v2(training_generator, datetime_generator)



model = Sequential([
    LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape),
    LSTM(units=50, activation='relu', return_sequences=True),
    LSTM(units=50, activation='relu'),
    Dense(units=1, activation='linear')
])
model.compile(optimizer='adam', loss='mae')

model.fit(filtered_training_generator_batch_list, epochs=1)


model.fit(training_generator, epochs=5)

#training 
train_predictions = model.predict(training_generator)
predicted_values = scaler_y.inverse_transform(train_predictions).reshape(-1)[:-pred_steps]

# "validate" the model
training_score = model.evaluate(training_generator)
print("Validation Loss:", training_score)


# visualisation

# Extract the true y values for the chosen sequence
true_values = list(df_y.iloc[:].values.flatten())[sequence_length+pred_steps:]

print("r2 loss:", str(r2_score(true_values, predicted_values)))
print("mse loss:", str(mean_squared_error(true_values, predicted_values)))
print("mae loss:", str(mean_absolute_error(true_values, predicted_values)))

# Create a time axis for plotting
time_steps = np.arange(len(predicted_values))

# Plot the true and predicted values
plt.plot(time_steps, true_values, label='True Values', marker='o', markersize=4, linewidth=1)
plt.plot(time_steps, predicted_values, label='Predicted Values', marker='x', markersize=4, linewidth=1)

plt.title('True vs Predicted Values for a Single Sequence')
plt.xlabel('Time Steps')
plt.ylabel('Y Values')
plt.legend()
plt.show()


print("hello")