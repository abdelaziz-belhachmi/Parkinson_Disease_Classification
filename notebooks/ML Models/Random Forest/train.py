from random import random
import pandas as pd
import numpy as np
from RandomForest import RandomForest
from sklearn.model_selection import train_test_split

file_path = "../../../data/raw/parkinson_dataset.csv"
data = pd.read_csv(file_path, header=1)

target='class'
X=data.drop(columns=[target], axis=1)
y=data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

clf = RandomForest(n_trees=1)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = accuracy(y_test, predictions)
print(acc)