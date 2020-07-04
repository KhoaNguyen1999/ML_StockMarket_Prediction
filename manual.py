import math
import pickle
import sys
import matplotlib
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import pandas as pd


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle("Model")# name title
        self.axes = fig.add_subplot(111)
        self.axes.set_ylabel(ylabel="Close Price")
        self.axes.set_xlabel(xlabel="Date")
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        sc = MplCanvas(self, width=16, height=8, dpi=100)  # change in here

        # Create our pandas DataFrame with some simple
        # data and headers.
        df = pd.read_csv(r'C:\Users\USER\Documents\Python files\StockMarketPrediction-master\usStock.csv')

        # Create a new dataframe with only close column
        data = df.filter(['Close'])
        # Convert dataframe to numpy array
        dataset = data.values
        # Get the number of rows to train the model on
        training_data_len = math.ceil(len(dataset) * .8)
        print('training_data_len ',training_data_len)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        print('scaled_data')
        print(scaled_data)

        # Create the training data set
        # Create the scaled training data set
        train_data = scaled_data[0:training_data_len, :]
        # Split the data into x_train and y_train dataset
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])
            if i <= 60:
                print('x_train ',x_train)
                print('y_train',y_train)
                print()

        # Convert x_train y_train to numpy array
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_train.shape
        print('x_train.shape', x_train.shape)

        # Create testing dataset
        # Create a new array containing scaled values from index
        test_data = scaled_data[training_data_len - 60:, :]
        # Create dataset x_test and y_test
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, 0])

        # Convert the data to numpy array
        x_test = np.array(x_test)

        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        joblib.dump(model,'trained_model.pkl')
        model_from_joblib=joblib.load('trained_model.pkl')
        print('save ',model_from_joblib)


        # Get module predicted price value
        #predictions = model.predict(x_test)
        predictions=model_from_joblib.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Get the root mean squarred error (RMSE)
        rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
        print('rmse',rmse)

        # plot the data
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions

        # df["Date"] = pd.to_datetime(df['Date'])
        # df.set_index('Date', inplace=True)
        # df.set_index('Close', inplace=False)
        # df=df['Close']

        # df.plot(train['Close'])
        # df.plot(valid[['Close', 'Predictions']])
        # df.plot.legend(['Train','Val','Predictions'])

        # plot the pandas DataFrame, passing in the
        # matplotlib Canvas axes.
        df.plot(ax=sc.axes)
        self.setCentralWidget(sc)
        self.show()


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec_()
