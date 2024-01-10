import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

Start = date.today() - timedelta(1095)
Start.strftime('%Y-%m-%d')
End = date.today() - timedelta(2)
End.strftime('%Y-%m-%d')

################################################################################################################################################

# Functions
def logger(data, extra = "\n"):
    if (True):
        print(data, extra)

def closing_price(ticker):
    Asset = pd.DataFrame(yf.download(ticker, start=Start,
      end=End))
    return Asset

def build_and_compile_model(input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=64,
                                return_sequences=True,
                                input_shape=input_shape))
    model.add(keras.layers.LSTM(units=64))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.001))

    return model

################################################################################################################################################

shouldLoadModel = True # change this to true to prevent creating new model and training the model.

from sklearn.preprocessing import MinMaxScaler

# stockName = "MSFT"
# stockName = "AAPL"
stockName = "TSLA"
# stockName = "GOOG"

# getting the dataset and cleaning
stockData = closing_price(stockName)
stockData.drop(columns=['Adj Close'], inplace=True)
dataset = stockData.copy()
dataset = dataset.dropna()

# normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
scaled_data = pd.DataFrame(scaled_data, index=dataset.index, columns=dataset.columns)
logger(scaled_data)

################################################################################################################################################

# Splitting the data into a training set and a test set.
# train_dataset = scaled_data.sample(frac=0.8, random_state=0)
# test_dataset = scaled_data.drop(train_dataset.index)
train_dataset, test_dataset = np.split(scaled_data, [int(.8*len(scaled_data))])

logger(train_dataset.head())
logger(test_dataset.head())


logger("training dataset count:")
logger(train_dataset.count(), '\n')
logger("test dataset count:")
logger(test_dataset.count())

################################################################################################################################################

# Split data into features and labels
train_features = train_dataset.copy().astype('float32')
test_features = test_dataset.copy().astype('float32')

train_labels = train_features.pop('Close')
test_labels = test_features.pop('Close')

################################################################################################################################################

if (shouldLoadModel == False):
    dnn_model = build_and_compile_model((train_features.shape[1],1))

    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=.02,
        epochs=100)
    dnn_model.save('models/' + stockName)
else:
    dnn_model = tf.keras.models.load_model("models/MSFT")

logger(dnn_model.summary())

################################################################################################################################################

test_predictions = dnn_model.predict(test_features).flatten()
test_predictions = pd.Series(test_predictions, index=test_labels.index)

################################################################################################################################################

scaledTrainData = train_features.copy()
scaledTrainData.insert(3, 'Close', train_labels)

scaledTestData = test_features.copy()
scaledTestData.insert(3, 'Close', test_labels)

scaledPredictionData = test_features.copy()
scaledPredictionData.insert(3, 'Close', test_predictions)

unscaledTrainData = scaler.inverse_transform(scaledTrainData)
unscaledTrainData = pd.DataFrame(unscaledTrainData, index=train_dataset.index, columns=train_dataset.columns)

unscaledTestData = scaler.inverse_transform(scaledTestData)
unscaledTestData = pd.DataFrame(unscaledTestData, index=test_dataset.index, columns=test_dataset.columns)

unscaledPredictionData = scaler.inverse_transform(scaledPredictionData)
unscaledPredictionData = pd.DataFrame(unscaledPredictionData, index=test_dataset.index, columns=test_dataset.columns)


plt.figure(1)
plt.xlabel("date")
plt.ylabel("closing value")
plt.plot(unscaledTrainData['Close'], color="b", label="training close value")
plt.plot(unscaledTestData['Close'], color="g", label="testing close value")
plt.plot(unscaledPredictionData['Close'], color="r", label="predicted close value")
plt.title("stock closing price")
plt.legend()

plt.figure(2)
plt.xlabel("date")
plt.ylabel("closing value")
plt.plot(train_labels, color="b", label="training close value")
plt.plot(test_labels, color="g", label="testing close value")
plt.plot(test_predictions, color="r", label="predicted close value")
plt.title("stock closing price")
plt.legend()

plt.show()
logger(test_labels.compare(test_predictions).tail())
