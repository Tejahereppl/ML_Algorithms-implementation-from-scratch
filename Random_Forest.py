from DecisionTree import DecisionTree
import numpy as np
from collections import Counter
from tqdm import tqdm
class RandomForest:
    def __init__(self,min_samples_split = 2,max_depth = 100,n_trees = 50 ,n_features = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.n_features = n_features
        self.trees =[]

    def fit(self,X,y):

        for _ in tqdm(range(self.n_trees)):
            dt = DecisionTree(max_depth=self.max_depth,min_samples_split=self.min_samples_split,n_features = self.n_features )
            X_sample,Y_sample = self.sampler(X,y)

            dt.fit(X_sample,Y_sample)
            self.trees.append(dt)

    def sampler(self,X,y):
        n_samples = X.shape[0]
        ft_indx = np.random.choice(n_samples,n_samples,replace=True)

        return X[ft_indx],y[ft_indx]
    
    def most_common_value(self,y):
        if len(y) == 0:
            raise ValueError("Cannot determine the most common value from an empty array.")
        count = Counter(y)
        return count.most_common(1)[0][0]
    

    def predict(self,X):
        predictions = np.array([tree.predict(X) for tree in self.trees ])
        predictions = predictions.T

        return np.array([self.most_common_value(pred) for pred in predictions])
    







