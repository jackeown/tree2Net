import numpy as np
import typing
from sklearn.tree import DecisionTreeClassifier

def tree2Net(tree, saturation=5):
	"""
	Calculates Weight Matrices and bias vectors
	for a neural network, given a trained sklearn tree.
	Inspired by "Initializing Neural Networks using Decision
	Trees" by Arunava Banerjee.  Code by John McKeown (2017)

	Saturation refers to how saturated you want neurons to be.
	Returned matrices and biases assume data is passed in in row form.
	Matrix multiplication thus happens with the row on the left and the matrix
	on the right, as this is what I've seen more commonly with deep learning.
	"""

	ls = tree.tree_.children_left
	rs = tree.tree_.children_right
	fs = tree.tree_.feature
	ts = tree.tree_.threshold
	vs = tree.tree_.value
	n_classes = tree.tree_.n_classes[0]
	n_features = tree.tree_.n_features
	n_nodes = len(ls)


	weights = []
	biases = []

	# First hidden layer has all inequalities
	hiddenWeights = []
	hiddenBiases = []
	for f,t in zip(fs,ts):
		# f < 0 means that this node is a leaf.
		if f >= 0:
			# add row instead of column and later transpose.
			hiddenWeights.append([-1*saturation if i==f else 0 for i in range(n_features)])
			hiddenBiases.append(saturation*t)


	# hidden matrix transposed and appended to list of all weights
	weights.append(np.array(hiddenWeights).T.astype("float32"))
	biases.append(np.array(hiddenBiases).astype("float32"))
	n_splits = len(hiddenWeights)

	# Second hidden layer has ANDs for each leaf of the decision tree.
	# Depth first enumeration of the tree in order to determine the AND by the path.
	hiddenWeights = []
	hiddenBiases = []

	path = [0]
	visited = [False for i in range(n_nodes)]

	# save classes for later ORing
	classes = []
	nodes = list(zip(ls,rs,fs,ts,vs))
	while path != []:
		i = path[-1]
		visited[i] = True
		l,r,f,t,v = nodes[i]

		if l == -1 and r == -1: # leaf node
			vec = [0 for _ in range(n_splits)]
			#Keep track of positive weights for calculating bias.
			numPositive = 0
			for j,p in enumerate(path[:-1]):
				numLeavesBeforeP = list(ls[:p]).count(-1)
				if path[j+1] in ls:
					vec[p-numLeavesBeforeP] = saturation
					numPositive += 1
				elif path[j+1] in rs:
					vec[p-numLeavesBeforeP] = -saturation
				else:
					print("Warning: tree2Net done messed up yo...good luck")
			classes.append(np.argmax(vs[i]))
			hiddenWeights.append(vec)
			hiddenBiases.append(-(saturation*numPositive-saturation/2))
			path.pop()

		elif not visited[l]:
			path.append(l)

		elif not visited[r]:
			path.append(r)

		else:
			path.pop()
			

	# hidden matrix transposed and appended to list of all weights
	weights.append(np.array(hiddenWeights).T.astype("float32"))
	biases.append(np.array(hiddenBiases).astype("float32"))


	# OR neurons from the preceding layer in order to get final classes.
	outputWeights = []
	outputBiases = []

	for c in range(n_classes):
		outputWeights.append([saturation if i==c else 0 for i in classes])
		outputBiases.append(-saturation/2)

	weights.append(np.array(outputWeights).T.astype("float32"))
	biases.append(np.array(outputBiases).astype("float32"))
	return (weights,biases)





def forest2Nets(forest,saturation=5,verbose=False):
	"""
	Calculates Weight Matrices and Bias Vectors
	for neural networks, given a trained sklearn forest.
	(One weights/biases pair for each tree)

	Inspired by "Initializing Neural Networks using Decision
        Trees" by Arunava Banerjee.  Code by John McKeown (2017)

	Saturation refers to how saturated you want neurons to be.
	Returned matrices and biases assume data is passed in in row form.
	Matrix multiplication thus happens with the row on the left and the matrix
	on the right, as this is what I've seen more commonly with deep learning.
	"""
	nets = []
	for i,tree in enumerate(forest.estimators_):
		nets.append(tree2Net(tree,saturation))
		if verbose:
			print("converting tree {}".format(i+1))
	# return [tree2Net(tree,saturation) for tree in forest.estimators_]
	return nets
