import scipy.io as sio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import nibabel as nib
import os
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import gc



matfn = 'train_feat.mat'
data = sio.loadmat(matfn)
X1 = data['feat']


X1 = np.nan_to_num(X1)

X = X1

y = data['trlabel']

y = np.ravel(y)

# Split the dataset in two 4:1 parts
X_train, X_test, y_train, y_test = train_test_split(
    X1, y, test_size=0.2, random_state=0)
clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2,
                             random_state=0)  # 5
clf = clf.fit(X, y)
clfname1 = 'model0.txt'

joblib.dump(clf, clfname1)