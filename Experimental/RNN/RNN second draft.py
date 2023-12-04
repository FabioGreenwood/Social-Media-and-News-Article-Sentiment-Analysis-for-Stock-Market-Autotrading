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



# Example multivariate time series data (replace this with your actual time series)
# Assuming two features in the time series


# Create random multivariate time series data
df_X = pd.read_csv(r"C:\Users\Fabio\OneDrive\Documents\Studies\Final Project\Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading\precalculated_assets\example_input.csv")
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


keras.utils.timeseries_dataset_from_array


sequence_length = 10
input_shape = (sequence_length, df_X.shape[1])
batch_size = 32


training_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        df_X.values,
        df_y.values,
        sequence_length,
        batch_size=1,
        shuffle=False
    )

datetime_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        df_X.index.values,
        df_y.values,
        sequence_length,
        batch_size=1,
        shuffle=False
    )

## remove sequences that straddle a day
#for id in range(len(training_generator)):
#    if not datetime_generator[id][0][0][0][:10] == datetime_generator[id][0][0][-1][:10]:
#        training_generator[id] = None, None

model = Sequential([
    LSTM(units=50, activation='linear', return_sequences=True, input_shape=input_shape),
    LSTM(units=50, activation='linear', return_sequences=True),
    LSTM(units=50, activation='linear'),
    Dense(units=1, activation='linear')
])
model.compile(optimizer='adam', loss='mae')
model.fit(training_generator, epochs=5)

#training 
train_predictions = model.predict(training_generator)
predicted_values = train_predictions.reshape(-1)[:-pred_steps]

"""# manual training check
train_predictions_manually_generated_input = df_X.iloc[-10:,:].values.reshape((1, sequence_length, df_X.shape[1]))
train_predictions_2 = model.predict(train_predictions_manually_generated_input)
"""

# "validate" the model
#validation_dataset = timeseries_dataset_from_array(df_X.values, df_y.values, sequence_length=1, sampling_rate=1, shuffle=False)
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