from sklearn.neural_network import MLPClassifier as mlp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as rcf
from sklearn.metrics import confusion_matrix
import numpy as np

data = pd.read_csv('/mnt/d/Work/Acad/BTP/data/trainGreenBit/fractal_feature.csv',header=None)
# data2 = pd.read_csv('/mnt/d/Work/Acad/BTP/data/trainGreenBit/feature.csv',header=None)
# data = data.join(data2.iloc[:,2:], lsuffix='_caller', rsuffix='_other')
data.dropna(inplace=True)
X_train = data.iloc[:,2:]
y_train = data.iloc[:,1]
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.05, random_state=42)

testdata = pd.read_csv('/mnt/d/Work/Acad/BTP/data/testGreenBit/fractal_feature.csv',header=None)
# testdata2 = pd.read_csv('/mnt/d/Work/Acad/BTP/data/testGreenBit/feature.csv',header=None)
# testdata =  testdata.join(testdata2.iloc[:,2:], lsuffix='_caller', rsuffix='_other')
testdata.dropna(inplace=True)
X = testdata.iloc[:,2:]
y = testdata.iloc[:,1]
X = scaler.transform(X)

clf = mlp(max_iter = 1000)
clf = clf.fit(X_train,y_train)
print('Validation Acc: ', clf.score(X_train,y_train))
print('Test Acc: ', clf.score(X,y))

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

97.27% val MLPC
82.66% test MLPC

93.64% val RFC
79.05% test RFC

Feature Importance:
7 17 18 8 20
ocl_mean, Valley(8), Ridge(1), ocl_var, Ridge(4)

with improved masking

100% val
84.16% test

Feature Importance
19, 7, 20, 11, 18

Confusion Matrix for Test
[[1770  270]
 [ 322 1376]]

2. DigitalPersona
98.18% val MLPC
81.96% test MLPC

96.36% val RFC
81.05% test RFC

Feature Importance:
0 7 8 18 17
num_minutiae ocl_mean, ocl_var Ridge(1), Valley(8)

with improved masking

99.99% val
82.53% test

Feature Importance
0, 8, 7, 19, 12

Confusion Matrix for Test
[[1592  436]
 [ 214 1478]]

3. Orcathus
0.915 val MLPC
0.8359 test MLPC

100 val RFC
100 test RFC

Feature Importance:
 0  7 17 15  8
 num_minutiae ocl_mean, ocl_var Ridge(1), Valley(8)

 with improved masking

100% val
85.58% test

Feature Importance
0, 7, 8, 18, 14

Confusion Matrix for Test
[[1741  277]
 [ 259 1441]]
   
"""