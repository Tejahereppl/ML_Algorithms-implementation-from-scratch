from DecisionTree import DecisionTree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

le = LabelEncoder()
df = pd.read_csv("updated_pollution_dataset.csv")
y = le.fit_transform(df["Air Quality"])

X = df.drop(columns = ["Air Quality"]).to_numpy()



X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2)

tree = DecisionTree()
tree.fit(X_train,Y_train)
y_pred = tree.predict(X_test)

def accuracy(y_pred,y_test):
    return np.sum(y_pred==y_test)/len(y_test)

print(accuracy(y_pred,Y_test))




