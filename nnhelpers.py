import numpy as np
def oneHot(y,n):
	"""Converts a label to a one-hot encoding"""
	return np.array([1 if y==i else 0 for i in range(n)])

def sigmoid(x):
	"""Simple S-shaped Activation Function with range (0,1)"""
	return 1/(1+np.exp(-x))

def applyNetwork(weights,biases,input):
	"""Computes sigmoid(wx+b) for each layer and outputs the final layer.
	This means 'weights' is a list of matrices and 'biases is a list of 'vectors'
	"""
	n_layers = len(weights)
	layer = input
	for i in range(n_layers):
		layer = sigmoid(np.dot(layer,weights[i])+biases[i])
	return layer
