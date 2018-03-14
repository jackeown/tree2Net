import numpy as np
from sklearn.datasets import load_iris,load_digits
from sklearn import tree
from tree2Net import tree2Net
from nnhelpers import *

data = load_iris()
xs = data.data
ys = data.target

clf = tree.DecisionTreeClassifier()
clf.fit(xs,ys)

weights,biases = tree2Net(clf,saturation=100)
out = applyNetwork(weights,biases,xs)

print("xs: {}".format(xs))
print("ys: {}".format(ys))
print("tree: {}".format(clf.predict(xs)))
print("network: {}".format(out))
