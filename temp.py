import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

df=pd.read_csv(r"C:\Users\U310\Downloads\archive\train.csv")
test=pd.read_csv(r"C:\Users\U310\Downloads\archive\test.csv")

# =============================================================================
# corrmat=df.corr()
# plt.subplots(figsize=(12,10))
# sns.heatmap(corrmat, vmax=0.8, square=True, annot=True, annot_kws={'size':8})
# =============================================================================

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

yt=np.array(df['price_range'])
xt=df.drop(['price_range'], axis=1)
xt=np.array(xt)

scaler=MinMaxScaler()
xt=scaler.fit_transform(xt)

xtrain,xtest,ytrain,ytest=train_test_split(xt, yt, test_size=0.2, random_state=42)

#Linear SVM을 위한 적절한 C값  검색
#분류:SVC, 회귀(예측):SVR

# =============================================================================
# scores = []
# for thisC in [*range(1,100)]:
#     svc=SVC(kernel='linear',C=thisC)
#     model=svc.fit(xtrain,ytrain)
#     scoreTrain=model.score(xtrain,ytrain)
#     scoreTest=model.score(xtest,ytest)
#     print("선형 SVM : C:{}, training score:{:2f}, test score:{:2f}".format
#           (thisC,scoreTrain, scoreTest))
#     scores.append([scoreTrain, scoreTest])
# =============================================================================

from sklearn.model_selection import GridSearchCV
param={'C':[1,5,10,20,40,100],
      'gamma':[.1, .25, .5, 1]}
GS=GridSearchCV(SVC(kernel='rbf'),param, cv=5)
GS.fit(xtrain, ytrain)
print(GS.best_params_)
print(GS.best_score_)

test=test.drop(['id'],axis=1)
test.head()
testmat=np.array(test)
test=scaler.fit_transform(test)
#test(DF -> array)
model=SVC(kernel='rbf', C=5, gamma=.1).fit(xtrain, ytrain)

prediction=model.predict(test)
pred=pd.DataFrame(prediction)
pred