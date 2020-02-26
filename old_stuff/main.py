import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris,load_digits,load_boston,load_breast_cancer
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from tree2Net import tree2Net
from MLPRegressor import MLPRegressor
from copy import deepcopy
from nnhelpers import *



# This file is an example of using tree2nn to make a neural network from an sklearn decision tree.
# After this, it uses MLPRegressor (A tensorflow implementation I wrote) to train the net further
# then it prints the accuracy of all these models along with another mlp for comparison.




# print(dir(datasets))
# data = load_iris()
# data = load_breast_cancer()
data = load_digits()

xs = data.data
ys = data.target
scaler = StandardScaler()
scaler.fit(xs)
xs = scaler.transform(xs)
print(xs[0],ys[0])

indices = np.random.choice(len(xs),len(xs))
xs,ys = (xs[indices],ys[indices])

nInputs = xs.shape[-1]
nOutputs = len(set(ys))

k_folds = 10
pos = lambda i:i*(len(xs)//k_folds)
folds = [(xs[pos(i):pos(i+1)],ys[pos(i):pos(i+1)]) for i in range(k_folds)]

accuracies = []
for leaveOut in range(k_folds):
	trainXs, trainYs = zip(*[fold for i,fold in enumerate(folds) if i!=leaveOut])
	trainXs,trainYs = (np.concatenate(trainXs),np.concatenate(trainYs))
	trainYs_oneHot = np.array([oneHot(y,nOutputs) for y in trainYs])

	testXs, testYs = folds[leaveOut]

	clf = tree.DecisionTreeClassifier()
	clf.fit(trainXs,trainYs)
	weights, biases = tree2Net(clf,saturation=5.0)
	i=0
	for w,b in zip(weights,biases):
		wDev = 1/np.sqrt(weights[i].shape[0])
		w[w==0] = np.random.normal(0,wDev,len(w[w==0]))
		b[b==0] = np.random.normal(0,wDev,len(b[b==0]))
		i += 1

	hparams = {
		"silent":False,
		"earlyStopping":True,
		"numEpochs":1000,
		"batchSize":len(xs)/10,
		"learningRate":0.01,
		"numInputs":nInputs,
		"numOutputs":nOutputs,
		"layers":[w.shape[-1] for w in weights[:-1]],
		"dropoutProbability":1.0,
		"validationPercent":0.1
	}

	hparamsPreTrained = deepcopy(hparams)
	hparamsPreTrained["weights"] = weights
	hparamsPreTrained["biases"] = biases

	print("TRAIN TREE MLP")
	treeNetTrained = MLPRegressor(hparamsPreTrained)
	treeNetTrained.fit(trainXs,trainYs_oneHot)

	print("TRAIN NORMAL MLP")
	mlp = MLPRegressor(hparams)
	mlp.fit(trainXs,trainYs_oneHot)

	treeOut = clf.predict(testXs)
	treeNetOut = np.argmax(applyNetwork(weights,biases,testXs),axis=1)
	treeNetTrainedOut = np.argmax(treeNetTrained.predict(testXs),axis=1)
	mlpOut = np.argmax(mlp.predict(testXs),axis=1)


	def accuracy(x,y):
		return sum(x==y)/len(x-y)

	accuracyTree = accuracy(testYs,treeOut)
	accuracyTreeNet = accuracy(testYs,treeNetOut)
	accuracyTreeNetTrained = accuracy(testYs,treeNetTrainedOut)
	accuracyMlp = accuracy(testYs,mlpOut)

	accuracies.append((accuracyTree,accuracyTreeNet,accuracyTreeNetTrained,accuracyMlp))

treeAccuracies,treeNetAccuracies,treeNetTrainedAccuracies,mlpAccuracies = zip(*accuracies)

print("tree accuracy: {}".format(np.mean(treeAccuracies)))
print("treeNet accuracy: {}".format(np.mean(treeNetAccuracies)))
print("treeNetTrained accuracy: {}".format(np.mean(treeNetTrainedAccuracies)))
print("mlp accuracy: {}".format(np.mean(mlpAccuracies)))

# print("tree accuracy: {}".format(treeAccuracies))
# print("treeNet accuracy: {}".format(treeNetAccuracies))
# print("treeNetTrained accuracy: {}".format(treeNetTrainedAccuracies))
# print("mlp accuracy: {}".format(mlpAccuracies))


for i in range(3):
	print("-"*100)
	print("weights before: {}".format(weights[i]))
	print("weights after: {}".format(treeNetTrained.getWeights()[i]))
