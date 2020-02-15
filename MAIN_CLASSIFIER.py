from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('/mnt/d/Work/Acad/BTP/data/trainGreenBit/feature.csv')
data.dropna(inplace=True)
clf = SVC(gamma='auto')
X = data.iloc[:,2:-1]

y = data.iloc[:,1]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
"""
96.33% validation acc.
20 18 17 16 3 15 7 0 19 2
Ridge(4), Ridge(1),Valley(8),Valley(7),Median,Valley(6),Ocl_mean,num_minutiae,Ridge(2),Entropy

"""