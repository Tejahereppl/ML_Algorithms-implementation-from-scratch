from DecisionTree import DecisionTree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Random_Forest import RandomForest
import numpy as np
import pandas as pd

le = LabelEncoder()
df = pd.read_csv("updated_pollution_dataset.csv")
y = le.fit_transform(df["Air Quality"])

X = df.drop(columns = ["Air Quality"]).to_numpy()



X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2)

decision_tree = DecisionTree()
Random_forest = RandomForest()
decision_tree.fit(X_train,Y_train)
Random_forest.fit(X_train,Y_train)
y_pred1 = decision_tree.predict(X_test)
y_pred2 = Random_forest.predict(X_test)

def accuracy(y_pred,y_test):
    return np.sum(y_pred==y_test)/len(y_test)

print("Accuracy for Decision Tree" + str(accuracy(y_pred1,Y_test)))
print("Accuracy for Random Forest" + str(accuracy(y_pred2,Y_test)))




