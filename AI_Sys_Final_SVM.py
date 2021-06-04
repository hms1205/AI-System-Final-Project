from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# train.csv, test.csv PATH
df=pd.read_csv(r"train.csv")
test=pd.read_csv(r"test.csv")

yt=np.array(df['price_range'])
xt=df.drop(['price_range'], axis=1)
xt=np.array(xt)

scaler=MinMaxScaler()
xt=scaler.fit_transform(xt)

sum = 0

for i in range(100):

    x_train,x_test,y_train,y_test = train_test_split(xt, yt, test_size=0.2, random_state=42)

    s = svm.SVC(C = 20, gamma = 0.01)
    s.fit(x_train, y_train)

    res = s.predict(x_test)

    conf = np.zeros((4,4))
    for i in range(len(res)):
        conf[res[i]][y_test[i]] += 1
    print(conf)

    correct = 0
    for i in range(4):
        correct += conf[i][i]
    accuracy = correct/len(res)
    print("Accuracy is", accuracy*100, "%. ")

    sum += accuracy
    
avg = sum/100

print("Average is", round(avg*100, 2), "%. ")
