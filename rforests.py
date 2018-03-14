import numpy as np
from scipy import stats

from MLPRegressor import MLPRegressor
from copy import deepcopy
from sklearn.datasets import load_iris,load_digits
from sklearn.ensemble import RandomForestClassifier
from tree2Net import *
from nnhelpers import *

# data = load_iris()
data = load_digits()


indices = np.random.choice(len(data.data),9*len(data.data)//10)
indicesComp = [i for i in range(len(data.data)) if i not in indices]

trainXs = data.data[indices]
testXs = data.data[indicesComp]

trainYs = data.target[indices]
n_outputs = len(set(trainYs))

trainYsOneHot = np.array([tuple(oneHot(y,n_outputs)) for y in trainYs])
testYs = data.target[indicesComp]
testYsOneHot = np.array([tuple(oneHot(y,n_outputs)) for y in testYs])


clf = RandomForestClassifier(n_estimators = 30)
clf = clf.fit(trainXs, trainYs)


networks = forest2Nets(clf,saturation=2.5)

def accuracy(out,ys):
	return np.mean(np.argmax(out,axis=1) == np.argmax(ys,axis=1))
	# return np.mean(1-(out-ys)**2)




baseParams = {
	"silent":False,
	"earlyStopping":False,
	"numEpochs":400,
	"batchSize":len(indices),
	"learningRate":0.01,
	"numInputs":np.array(trainXs).shape[-1],
	"numOutputs":len(set(trainYs)),
	# "layers":[w.shape[-1] for w in weights[:-1]],
	"dropoutProbability":1.0,
	"validationPercent":0.15
}




outs = []
trainedOuts = []
for i in range(len(networks)):
	print("#"*50 + " NETWORK " + str(i) + " " + "#"*50)
	weights = networks[i][0]
	biases = networks[i][1]
	outs.append(applyNetwork(weights,biases,testXs))

	hparams = deepcopy(baseParams)
	hparams["layers"] = [w.shape[-1] for w in weights[:-1]]
	hparams["weights"] = weights
	hparams["biases"] = biases

	model = MLPRegressor(hparams)
	model.fit(trainXs,trainYsOneHot)
	trainedOuts.append(model.predict(testXs))

print("done training!")

treeOut = np.array([oneHot(y,n_outputs) for y in clf.predict(testXs)])
# nlpOut = stats.mode(outs)[0][0]
# trainedNlpOut = stats.mode(trainedOuts)[0][0]
nlpOut = np.round(np.mean(outs,axis=0))
trainedNlpOut = np.round(np.mean(trainedOuts,axis=0))


print("Forest: {}".format(accuracy(treeOut,testYsOneHot)))
print("Forest2Nets: {}".format(accuracy(nlpOut,testYsOneHot)))
print("TrainedForest2Nets: {}".format(accuracy(trainedNlpOut,testYsOneHot)))


print("#"*100)
print(treeOut)
print("#"*100)
print(nlpOut)
print("#"*100)
print(trainedNlpOut)
