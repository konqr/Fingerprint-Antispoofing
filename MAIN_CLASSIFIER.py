from sklearn.neural_network import MLPClassifier as mlp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as rcf
from sklearn.metrics import confusion_matrix
import numpy as np

#data = pd.read_csv('/mnt/d/Work/Acad/BTP/data/trainGreenBit/feature.csv',header=None)
#data = pd.read_csv('/mnt/d/Work/Acad/BTP/data/trainGreenBit/fractal_feature_enh5.csv',header=None)
data = pd.read_csv('/mnt/d/Work/Acad/BTP/data/trainGreenBit/feature_patches.csv',header=None)
# data = data.join(data2.iloc[:,2:], lsuffix='_caller', rsuffix='_other')
data.dropna(inplace=True)
X_train = data.iloc[:,2:]
y_train = data.iloc[:,1]
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.1, random_state=42)

testdata = pd.read_csv('/mnt/d/Work/Acad/BTP/data/testGreenBit/feature_patches.csv',header=None)
# testdata = pd.read_csv('/mnt/d/Work/Acad/BTP/data/testGreenBit/fractal_feature.csv',header=None)
# testdata2 = pd.read_csv('/mnt/d/Work/Acad/BTP/data/testGreenBit/feature.csv',header=None)
# testdata =  testdata.join(testdata2.iloc[:,2:], lsuffix='_caller', rsuffix='_other')
testdata.dropna(inplace=True)
X = testdata.iloc[:,3:]
y = testdata.iloc[:,2]
X = scaler.transform(X)

clf = mlp(hidden_layer_sizes=(100,20, ), max_iter = 1000,verbose=1)
clf = clf.fit(X_train,y_train)
print('Validation Acc: ', clf.score(X_val,y_val))
print('Test Acc: ', clf.score(X,y))

y_test_prob = pd.DataFrame(clf.predict_proba(X))
sample = pd.DataFrame(testdata.iloc[:,1]).reset_index()
joined = sample.join(y_test_prob,lsuffix='a')
prob_pred = joined.groupby(['1a']).mean()
joined = sample.join(pd.get_dummies(y).reset_index(),lsuffix='a')
prob_true = joined.groupby(['1a']).mean()

RCF = rcf()
RCF = RCF.fit(X_train,y_train)
print(RCF.score(X_train,y_train))
print(RCF.score(X,y))
print(np.argsort(-1*RCF.feature_importances_))

y_o_val = clf.predict(X_train)
y_o_test = clf.predict(X)
print("Validation confusion_matrix")
print(confusion_matrix(y_train,y_o_val))
print("Test confusion_matrix")
print(confusion_matrix(y,y_o_test))

# for k in range(34):
#      gel = gel.append(testdata.iloc[110*k:110*k+20,:], ignore_index=True)

"""
RESULTS:

1. GreenBit

100% val
84.16% test

Feature Importance
19, 7, 20, 11, 18

Confusion Matrix for Test
[[1770  270]
 [ 322 1376]]

 FSA: 270/total = 7.2%
 FLR: 322/total = 8.6%

2. DigitalPersona

99.99% val
82.53% test

Feature Importance
0, 8, 7, 19, 12

Confusion Matrix for Test
[[1592  436]
 [ 214 1478]]

 FSA = 9.2%
 FLR = 5.8%

3. Orcathus

100% val
85.58% test

Feature Importance
0, 7, 8, 18, 14

Confusion Matrix for Test
[[1741  277]
 [ 259 1441]]

 FSA = 7.4%
 FLR = 6.9%

 Total FSA: 7.9%
 Total FLR: 7.1%
   
"""