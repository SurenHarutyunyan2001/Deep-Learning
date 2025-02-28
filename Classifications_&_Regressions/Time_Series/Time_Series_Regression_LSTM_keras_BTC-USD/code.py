import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
from keras.src.layers import *
from keras.src.models import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.losses import MeanSquaredError
from keras.src.metrics import *
from keras.src.optimizers import Adam
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# read by default 1st sheet of an excel file
dataframe1 = pd.read_excel('2.xlsx')

data = dataframe1.to_numpy()
data = np.flip(data,axis=0)
#print(data)

test = data[:,0:1]
train = data[:,0:4]
#print(test)

scaler = MinMaxScaler(feature_range=(0,1))
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)

time_step = 10
x,y = [],[]

for i in range(len(train)-time_step):
    row = [a for a in train[i:i+time_step]]
    x.append(row)
    lable = [a for a in test[i+time_step]]
    y.append(lable)
x = np.array(x)
y = np.array(y) 

train_len = int(len(x)*0.9)
x_train, y_train = x[:train_len], y[:train_len]
x_test, y_test = x[train_len:], y[train_len:]
y_train = y_train.reshape(int(len(y_train)),-1)
y_test = y_test.reshape(int(len(y_test)),-1)

model = Sequential()
model.add(LSTM(50,return_sequences = True,input_shape = (time_step,4)))
model.add(LSTM(50,return_sequences = False))

model.add(Dense(25,activation="relu"))
model.add(Dense(1,activation="linear"))
model.compile(optimizer = 'adam',loss = 'mse',metrics = ['mae'])
model.fit(x_train,y_train,epochs=200,batch_size=100)  #epochs=100,batch_size=100
model.summary()

pred_x_test = model.predict(x_test)
pred_x_test = scaler.inverse_transform(pred_x_test)
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

y_train_len = list(range(len(y_train)))
y_test_len = list(range(len(y_test)))
y_test_len = list(range(len(y_train),len(y_test)+len(y_train)))

plt.plot(y_train_len,y_train.flatten())
plt.plot(y_test_len,y_test.flatten())
plt.plot(y_test_len,pred_x_test,color = "r")
plt.show()

x_new = train[len(train)-time_step:]
x_new = np.expand_dims(x_new,axis=0)
x_new_pred = model.predict(x_new)
x_new_pred = scaler.inverse_transform(x_new_pred)

print(y_test[-1],"last price")

print(x_new_pred,"pragnoz")




print("aaaaaaaaaaaaa")