import numpy as np
from tqdm import tqdm
from collections import Counter
class Node:
    def __init__(self,feature = None,threshold = None,left = None,right = None,*,value = None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
    
    def is_leaf_node(self):
        #My code
        '''if self.value!=None:
            return True
        return False'''
        #Optimised one ig
        return self.value is not None

class DecisionTree:
    def __init__(self,min_samples_split=2,max_depth=100,n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
    
    def fit(self,X,y):
        
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self.grow_tree(X,y)

    def grow_tree(self,X,y,depth = 0):
        n_samples = X.shape[0]
        n_feats = X.shape[1]
        n_labels = len(np.unique(y))

        if depth>=self.max_depth or n_samples<=self.min_samples_split or n_labels ==1 :
            leaf_value = self.most_common_value(y)
            return Node(value = leaf_value )
        
        feat_idxs = np.random.choice(n_feats,self.n_features,replace = False)
        
        best_feature,best_threshold = self.best_split(X,y,feat_idxs)

        left_idxs,right_idxs = self.split(X[:,best_feature],best_threshold)

        left = self.grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right = self.grow_tree(X[right_idxs,:],y[right_idxs],depth+1)

        return Node(best_feature,best_threshold,left,right)
    
    def best_split(self,X,y,feat_idxs):
        best_gain = -1
        split_idx,split_thr = None,None

        for f_idx in feat_idxs:
            X_column = X[:,f_idx]
            thresholds = np.unique(X[:,f_idx])

            for thr in thresholds:
                gain = self.information_gain(X_column,y,thr)

                if gain>best_gain:
                    best_gain = gain
                    split_idx = f_idx
                    split_thr = thr
        return split_idx,split_thr

    def information_gain(self,X_column,y,thr):
        #parent Entropy

        parent_entropy = self.entropy(y)
        #find children

        left_idxs,right_idxs = self.split(X_column,thr)

        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0
        # weight average entropy
        n = len(y)
        n_l,n_r = len(left_idxs),len(right_idxs)
        e_l,e_r = self.entropy(y[left_idxs]),self.entropy(y[right_idxs])

        weighted_avg = (n_l/n)*e_l + (n_r/n)*e_r
        #information gain
        IG = parent_entropy - weighted_avg

        return IG

        
    
    def entropy(self,y):
        cnt = np.bincount(y)
        ps = cnt/len(y)
        return -np.sum([p*np.log2(p) for p in ps if p>0])
    
    def split(self,X_column,thr):
        left_idxs = np.argwhere(X_column<=thr).flatten()
        right_idxs = np.argwhere(X_column>thr).flatten()
        return left_idxs,right_idxs


    def predict(self,X):
        return np.array([self.traverse_tree(x,self.root) for x in tqdm(X)])
    
    def traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature]<=node.threshold:
            return self.traverse_tree(x,node.left)
        return self.traverse_tree(x,node.right)



        
    def most_common_value(self,y):
        if len(y) == 0:
            raise ValueError("Cannot determine the most common value from an empty array.")
        count = Counter(y)
        return count.most_common(1)[0][0]
        




    

