from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# train.csv, test.csv PATH
df=pd.read_csv(r"C:\Users\U310\Downloads\archive\train.csv")
test=pd.read_csv(r"C:\Users\U310\Downloads\archive\test.csv")

yt=np.array(df['price_range'])
xt=df.drop(['price_range'], axis=1)
xt=np.array(xt)

scaler=MinMaxScaler()
xt=scaler.fit_transform(xt)

# x_train,x_test,y_train,y_test = train_test_split(xt, yt, test_size=0.2, random_state=42)

(x_train, y_train) = xt
(x_test, y_test) = yt
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = xtrain/255.0
x_test = xtest/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

n_input = 784
n_hidden1 = 1024
n_hidden2 = 512
n_hidden3 = 512
n_hidden4 = 512
n_output = 10

mlp = Sequential()
mlp.add(Dense(units = n_hidden1, activation = 'tanh', input_shape=(n_input,), kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
mlp.add(Dense(units = n_hidden2, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
mlp.add(Dense(units = n_hidden3, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
mlp.add(Dense(units = n_hidden4, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
mlp.add(Dense(units = n_output, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))

mlp.compile(loss = 'mse', optimizer = Adam(learning_rate = 0.1), metrics = ['accuracy'])
hist = mlp.fit(x_train, y_train, batch_size = 128, epochs = 30, validation_data = (x_test, y_test), verbose = 2)

res = mlp.evaluate(x_test, y_test, verbose = 0)
print("Accuracy is", res[1]*100)