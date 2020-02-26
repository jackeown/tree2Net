from tree2Net import tree2Net
import torch
from nnhelpers import oneHot

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets




class TreeModule(torch.nn.Module):

    def __init__(self, tree, saturation=5):
        super().__init__()
        weights, biases = tree2Net(tree, saturation)
        self.w0 = torch.nn.Parameter(torch.tensor(weights[0]))
        self.w1 = torch.nn.Parameter(torch.tensor(weights[1]))
        self.w2 = torch.nn.Parameter(torch.tensor(weights[2]))
        
        self.b0 = torch.nn.Parameter(torch.tensor(biases[0]))
        self.b1 = torch.nn.Parameter(torch.tensor(biases[1]))
        self.b2 = torch.nn.Parameter(torch.tensor(biases[2]))

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        a0 = self.sigmoid((x @ self.w0) + self.b0)
        a1 = self.sigmoid((a0 @ self.w1) + self.b1)
        a2 = (a1 @ self.w2) + self.b2
        return a2
    

iris = datasets.load_iris()
xs = torch.tensor(iris.data, dtype=torch.float)
ys = torch.tensor(iris.target)

# train an sklearn decision tree...
tree = DecisionTreeClassifier()
tree.fit(xs,ys)

# convert it to a neural net...
myTreeNet = TreeModule(tree, 5)

# train the neural net further...
lossF = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(myTreeNet.parameters())

for i in range(10000):
    opt.zero_grad()

    # xs, ys = getBatch()
    pred = myTreeNet(xs)
    loss = lossF(pred, ys)

    loss.backward()
    opt.step()

    if i % 1000 == 0:
        print(f"loss: {loss}")


