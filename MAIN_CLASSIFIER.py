from sklearn.neural_network import MLPClassifier as mlp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as rcf
from sklearn.metrics import confusion_matrix

data = pd.read_csv('/mnt/d/Work/Acad/BTP/data/trainDigitalPersona/feature3.csv',header=None)
data.dropna(inplace=True)
X_train = data.iloc[:,2:]
y_train = data.iloc[:,1]
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.05, random_state=42)

testdata = pd.read_csv('/mnt/d/Work/Acad/BTP/data/testDigitalPersona/feature3.csv',header=None)
testdata.dropna(inplace=True)
X = testdata.iloc[:,2:]
y = testdata.iloc[:,1]
X = scaler.transform(X)

clf = mlp(max_iter = 1000,early_stopping=True)
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

2. DigitalPersona
98.18% val MLPC
81.96% test MLPC

96.36% val RFC
81.05% test RFC

Feature Importance:
0 7 8 18 17
num_minutiae ocl_mean, ocl_var Ridge(1), Valley(8)

3.   
"""