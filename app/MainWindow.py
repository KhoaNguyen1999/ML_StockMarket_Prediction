# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import matplotlib
import matplotlib.pyplot as plt
import pyqtgraph as pg
import numpy as np
import pandas as pd
import math
import csv
import os.path, time
import datetime as dtime
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.externals import joblib
from numpy import loadtxt
from keras.models import load_model
import keras
import h5py

matplotlib.use('Qt5Agg')
plt.style.use('bmh')


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100,filePath=""):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle("Close Price History")# name title
        self.axes = fig.add_subplot(111)
        self.axes.set_ylabel(ylabel="Close Price Price USD")
        self.axes.set_xlabel(xlabel="Date")
        FigureCanvasQTAgg.__init__(self,fig)
        self.setParent(parent)
        self.plot_value_single(filePath)

    def plot_value_single(self,filePath):
        # Create our pandas DataFrame with some simple
        # data and headers.
        df = pd.read_csv(filePath)
        df["Date"] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.set_index('Close', inplace=False)
        #df=df['Close']
        df=pd.DataFrame(df[['Close']],columns=['Close'])

        # plot the pandas DataFrame, passing in the
        # matplotlib Canvas axes.
        df.plot(ax=self.axes)
        #HistoryPlotWindow.setCentralWidget(sc)

class MplCanvas_Predict(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100, filePath="", data = pd.core.frame.DataFrame, training_data_len = 1, predictions =np.ndarray):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle("Close Price History")# name title
        self.axes = fig.add_subplot(111)
        self.axes.set_ylabel(ylabel="Close Price Price USD")
        self.axes.set_xlabel(xlabel="Date")
        FigureCanvasQTAgg.__init__(self,fig)
        self.setParent(parent)
        self.plot_value(filePath,data, training_data_len, predictions)

    def plot_value(self,filePath,data,training_data_len, predictions):
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        # Create our pandas DataFrame with some simple
        # data and headers.
        df = pd.read_csv(filePath)
        dt=df["Date"]
        df["Date"] = pd.to_datetime(df['Date'])
        

        df_1= pd.DataFrame(train[['Close']])
        df_1['Date'] = dt
        #print(df_1)
        df_1["Date"] = pd.to_datetime(df_1['Date'])


        df_2 = pd.DataFrame(valid[['Close', 'Predictions']])
        df_2['Date'] = dt
        df_2["Date"] = pd.to_datetime(df_2['Date'])

        df = pd.concat([df_1,df_2])
        df.set_index('Date', inplace=True)
        df.set_index('Close', inplace=False)
        # plot the pandas DataFrame, passing in the
        # matplotlib Canvas axes.
        df.plot(ax=self.axes)
        #HistoryPlotWindow.setCentralWidget(sc)

class MplCanvas_Linear(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100, filePath = ''):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle("Close Price History")# name title
        self.axes = fig.add_subplot(111)
        self.axes.set_ylabel(ylabel="Close Price Price USD")
        self.axes.set_xlabel(xlabel="Date")
        FigureCanvasQTAgg.__init__(self,fig)
        self.setParent(parent)
        self.plot_value(filePath)

    def plot_value(self,filePath):
        df = pd.read_csv(filePath)
        dt=df["Date"]
        training_data_len = math.ceil(len(df) * .25)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date']=df['Date'].map(dtime.datetime.toordinal)
        #Create a variable to predict 'x' days out into the future
        future_days = training_data_len
        #Create a new column (the target or dependent variable) shifted 'x' units/days up
        df['Prediction'] = df[['Close']].shift(-future_days)
        X = np.array(df.drop(['Prediction'], 1))[:-future_days]
        y = np.array(df['Prediction'])[:-future_days]
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
        #Create the linear regression model
        lr = LinearRegression().fit(x_train, y_train)

        #Get the feature data, 
        #AKA all the rows from the original data set except the last 'x' days
        x_future = df.drop(['Prediction'], 1)[:-future_days]
        #Get the last 'x' rows
        x_future = x_future.tail(future_days) 
        #Convert the data set into a numpy array
        x_future = np.array(x_future)

        #Show the model linear regression prediction
        lr_prediction = lr.predict(x_future)
        #print(lr_prediction)

        #Visualize the data
        predictions = lr_prediction

        print("MyLength")
        print(len(predictions))
        #Plot the data
        valid =  df[X.shape[0]:]
        valid['Predictions'] = predictions #Create a new column called 'Predictions' that will hold the predicted prices
        
        myValid = pd.DataFrame[[valid[['Close','Predictions']]]]
        print(myValid)
        # Create our pandas DataFrame with some simple
        # data and headers.
        
        df_1= pd.DataFrame(df[['Close']])
        df_1['Date'] = dt
        df_1["Date"] = pd.to_datetime(df_1['Date'])


        df_2 = pd.DataFrame(valid[['Close', 'Predictions']])
        df_2['Date'] = dt
        df_2["Date"] = pd.to_datetime(df_2['Date'])

        df = pd.concat([df_1,df_2])
        df.set_index('Date', inplace=True)
        df.set_index('Close', inplace=False)
        # plot the pandas DataFrame, passing in the
        # matplotlib Canvas axes.
        df.plot(ax=self.axes)
        #HistoryPlotWindow.setCentralWidget(sc)

class MplCanvas_TreeDecision(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100, filePath = ''):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle("Close Price History")# name title
        self.axes = fig.add_subplot(111)
        self.axes.set_ylabel(ylabel="Close Price Price USD")
        self.axes.set_xlabel(xlabel="Date")
        FigureCanvasQTAgg.__init__(self,fig)
        self.setParent(parent)
        self.plot_value(filePath)

    def plot_value(self,filePath):
        df = pd.read_csv(filePath)
        dt=df["Date"]
        training_data_len = math.ceil(len(df) * .25)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date']=df['Date'].map(dtime.datetime.toordinal)
        #Create a variable to predict 'x' days out into the future
        future_days = training_data_len
        #Create a new column (the target or dependent variable) shifted 'x' units/days up
        df['Prediction'] = df[['Close']].shift(-future_days)
        X = np.array(df.drop(['Prediction'], 1))[:-future_days]
        y = np.array(df['Prediction'])[:-future_days]
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
        #Create the decision tree regressor model
        tree = DecisionTreeRegressor().fit(x_train, y_train)

        #Get the feature data, 
        #AKA all the rows from the original data set except the last 'x' days
        x_future = df.drop(['Prediction'], 1)[:-future_days]
        #Get the last 'x' rows
        x_future = x_future.tail(future_days) 
        #Convert the data set into a numpy array
        x_future = np.array(x_future)

        #Show the model tree prediction
        tree_prediction = tree.predict(x_future)

        #Visualize the data
        predictions = tree_prediction
        #Plot the data
        valid =  df[X.shape[0]:]
        valid['Predictions'] = predictions #Create a new column called 'Predictions' that will hold the predicted prices
        # Create our pandas DataFrame with some simple
        # data and headers.
        
        df_1= pd.DataFrame(df[['Close']])
        df_1['Date'] = dt
        df_1["Date"] = pd.to_datetime(df_1['Date'])


        df_2 = pd.DataFrame(valid[['Close', 'Predictions']])
        df_2['Date'] = dt
        df_2["Date"] = pd.to_datetime(df_2['Date'])

        df = pd.concat([df_1,df_2])
        df.set_index('Date', inplace=True)
        df.set_index('Close', inplace=False)
        # plot the pandas DataFrame, passing in the
        # matplotlib Canvas axes.
        df.plot(ax=self.axes)
        #HistoryPlotWindow.setCentralWidget(sc)



class Ui_MainWindow(object):
    def onCreate(self):
        # add global string to share selected option to multiple instance
        # data value to create plot based on dataset
        self.currentPredictDataType = "Close"
        self.dataFilePath = ''

    def openHistoryPlotWindow(self):
        self.historyWindow = QtWidgets.QMainWindow()
        self.ui = Ui_HistoryPlotWindow(self.currentPredictDataType, self.dataFilePath)
        self.ui.setupUi(self.historyWindow)
        self.historyWindow.show()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1436, 1139)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.browseGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.browseGroupBox.setGeometry(QtCore.QRect(30, 10, 671, 121))
        self.browseGroupBox.setObjectName("browseGroupBox")
        self.browseButton = QtWidgets.QPushButton(self.browseGroupBox)
        self.browseButton.setGeometry(QtCore.QRect(540, 60, 101, 41))
        self.browseButton.setObjectName("browseButton")
        self.selectedLabel = QtWidgets.QLabel(self.browseGroupBox)
        self.selectedLabel.setGeometry(QtCore.QRect(0, 20, 641, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.selectedLabel.setFont(font)
        self.selectedLabel.setObjectName("selectedLabel")
        self.datGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.datGroupBox.setGeometry(QtCore.QRect(10, 200, 1905, 1151))
        self.datGroupBox.setObjectName("datGroupBox")
        self.fileDataTableView = QtWidgets.QTableView(self.datGroupBox)
        self.fileDataTableView.setGeometry(QtCore.QRect(0, 30, 721, 341))
        self.fileDataTableView.setObjectName("fileDataTableView")

        self.plotHistoryGroupBox = QtWidgets.QGroupBox(self.datGroupBox)
        self.plotHistoryGroupBox.setGeometry(QtCore.QRect(760, 20, 651, 351))
        self.plotHistoryGroupBox.setObjectName("groupBox")
        self.predictDataGroupBox = QtWidgets.QGroupBox(self.datGroupBox)
        self.predictDataGroupBox.setGeometry(QtCore.QRect(0, 430, 721, 351))
        self.predictDataGroupBox.setObjectName("groupBox_2")
        self.predictDiagramGroupBox = QtWidgets.QGroupBox(self.datGroupBox)
        self.predictDiagramGroupBox.setGeometry(QtCore.QRect(750, 430, 651, 351))
        self.predictDiagramGroupBox.setObjectName("groupBox_3")

        self.lrPredictDiagramGroupBox = QtWidgets.QGroupBox(self.datGroupBox)
        self.lrPredictDiagramGroupBox.setGeometry(QtCore.QRect(1420,30,471,341))
        self.lrPredictDiagramGroupBox.setObjectName("groupBox_4")

        self.tdPredictDiagramGroupBox = QtWidgets.QGroupBox(self.datGroupBox)
        self.tdPredictDiagramGroupBox.setGeometry(QtCore.QRect(1420,430,471,341))
        self.tdPredictDiagramGroupBox.setObjectName("groupBox_5")
        #self.showPredictionButton = QtWidgets.QPushButton(self.centralwidget)
        #self.showPredictionButton.setGeometry(QtCore.QRect(1160, 140, 121, 41))
        #self.showPredictionButton.setObjectName("showPredictionButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1436, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.output_rd = QtWidgets.QTextBrowser(self.predictDataGroupBox)
        self.output_rd.setGeometry(QtCore.QRect(10, 25, 700, 300))
        self.output_rd.setObjectName("output_rd")


        #self.addDataToDataComboBox()
        # Initialize onCreate Variable
        self.onCreate()

        # Event
        self.addEvent()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.browseGroupBox.setTitle(_translate("MainWindow", "Choose Data File"))
        self.browseButton.setText(_translate("MainWindow", "BROWSE"))
        self.selectedLabel.setText(_translate("MainWindow", "No file selected"))
        self.datGroupBox.setTitle(_translate("MainWindow", "Display Data"))
        #self.showDataButton.setText(_translate("MainWindow", "Show Data History"))
        self.plotHistoryGroupBox.setTitle(_translate("MainWindow", "Plot History")) #group box
        self.predictDataGroupBox.setTitle(_translate("MainWindow", "Predict Data")) #group box 2
        self.predictDiagramGroupBox.setTitle(_translate("MainWindow", "Prediction Diagram")) #group box 3
        self.lrPredictDiagramGroupBox.setTitle(_translate("MainWindow", "Linear Regression")) #box 4
        self.tdPredictDiagramGroupBox.setTitle(_translate("MainWindow", "Tree Decision")) #box 5

        #self.showPredictionButton.setText(_translate("MainWindow", "Show Prediction"))

    def addEvent(self):
        #self.dataComboBoxEvent()
        self.browseEvent()
        #self.plotHistoryData()
        #self.browseButton_Test.clicked.connect(self.test)

    # region update_csv
    def getCsvLastModified(self):
        self.lastModFile = time.ctime(os.path.getmtime(self.dataFilePath))
        # print(time.ctime(os.path.getmtime(self.dataFilePath)))

    def printLogLastModifiedFile(self):
        self.logFile = self.dataFileName + '\t' + self.lastModFile

    def printToLogFile(self):
        with open('LOG.txt', 'a+') as f:
            f.write(self.logFile)

    def checkLog(self):
        with open('LOG.txt', 'r') as f:
            for line in f:
                if (line == self.logFile):
                    return False
        return True

    # endregion

    # region dataComboBox
    def dataComboBoxEvent(self):
        # default option for dataComboBox
        # currentSelected = "Close Price"

        # Connect dataComboBox to change selected option
        self.dataComboBox.currentTextChanged.connect(self.dataComboBoxClicked)

    def addDataToDataComboBox(self):
        # Add options
        self.dataComboBox.addItems(["Close Price", "Open Price", "Highest", "Lowest"])

    def dataComboBoxClicked(self, currentSelected):
        currentSelected = self.dataComboBox.currentText()
        self.currentPredictDataType = self.setCurrentPredictionType(currentSelected)
        #print(self.currentPredictDataType)

    def setCurrentPredictionType(self, currentSelected):
        return {
            'Close Price': 'Close',
            'Open Price': 'Open',
            'Highest': 'High',
            'Lowest': 'Low'
        }.get(currentSelected, 'Close')

    # endregion

    # region browse
    # Browse Event
    def browseEvent(self):
        self.setModelDataTableView()
        self.browseButton.clicked.connect(self.openFileDialog)

    # set model for fileDataTableView
    def setModelDataTableView(self):
        self.model = QtGui.QStandardItemModel(self.centralwidget)
        self.fileDataTableView.setModel(self.model)
        self.fileDataTableView.horizontalHeader().setStretchLastSection(True)

    # Create openfile dialog when browse button cliked
    def openFileDialog(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            # Get string of selected file
            str = self.listToString(filenames)
            # Change currentTextlabel with selected file path
            self.changeSelectedLabel(str)
            # Set dataFilePath
            self.dataFilePath = str
            self.dataFileName = Path(str).stem
            # Populate dataTableView with data in the file
            self.loadCsv(str)
            #760, 20, 651, 351
            sc = MplCanvas(MainWindow, width=6, height=3, dpi=100, filePath = str)  # change in here
            sc.move(800,250)
            sc.show()
            #print()
            self.test(str)
            self.Linear(str)
            self.Tree(str)
            #ui = Ui_MainWindow()
            #ui.setupUi(MainWindow)
            
            

    # Get full path as string  from openFileDialog
    def listToString(self, s):
        str1 = ""
        for word in s:
            str1 = str1 + word
        return str1

    # Change currentTextlabel with selected file path
    def changeSelectedLabel(self, text):
        self.selectedLabel.setText(text)

    # Populate dataTableView with data in the file
    def loadCsv(self, fileName):
        with open(fileName, "r") as fileInput:
            for row in csv.reader(fileInput):
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.model.appendRow(items)



    # endregion

    # region plot history data
    # def plotHistoryData(self):
    #     self.showDataButton.clicked.connect(self.openHistoryPlotWindow)

    # endregion

    # region test_def
    def test(self,str):
        df = pd.read_csv(str)
        #Create a new dataframe with only close column
        data = df.filter(['Close'])
        #Convert dataframe to numpy array
        dataset = data.values
        #Get the number of rows to train the model on
        training_data_len = math.ceil(len(dataset) * .8)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        train_data = scaled_data[0:training_data_len, :]
        #Split the data into x_train and y_train dataset
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i ,0])
            y_train.append(train_data[i,0])
            if i <= 60:
                print(x_train)
                print(y_train)
        x_train, y_train = np.array(x_train) , np.array(y_train)
        x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        x_train.shape
        model = Sequential()
        model.add(LSTM(50, return_sequences= True ,input_shape = (x_train.shape[1],1)))
        model.add(LSTM(50, return_sequences = False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        #model.fit(x_train, y_train, batch_size=1, epochs=1)

        #joblib.dump(model,'trained_model.pkl')
        model_from_joblib=joblib.load('trained_model.pkl')
        test_data = scaled_data[training_data_len - 60: , :]
        #Create dataset x_test and y_test
        x_test= []
        y_test= dataset[training_data_len:, :]
        for i in range(60,len(test_data)):
            x_test.append(test_data[i-60:i,0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
        predictions = model_from_joblib.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        #print(valid)
        self.output_rd.append(valid.to_string())
        predictSc= MplCanvas_Predict(self.predictDiagramGroupBox, width=6, height=3, dpi=100, filePath = str,data = df.filter(['Close']), training_data_len = training_data_len, predictions = predictions)
        predictSc.move(15,25)
        predictSc.show()
# endregion

#region ???
    def Linear(self, filePath):
        linearSc = MplCanvas_Linear(self.lrPredictDiagramGroupBox, width=4, height=3, dpi=100, filePath = self.dataFilePath)
        linearSc.move(15,25)
        linearSc.show()

    def Tree(self, filePath):
     	tree = MplCanvas_TreeDecision(self.tdPredictDiagramGroupBox, width=4, height=3, dpi=100, filePath = self.dataFilePath)
     	tree.move(15,25)
     	tree.show()

        
#endregion

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
