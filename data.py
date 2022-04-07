import torch as t
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import tensorflow
from keras.datasets.mnist import load_data

from mlp import Mlp

(trainX, trainY), (testX, testY) = load_data()
trainX = (t.tensor(trainX, dtype=t.float) / 127.5 - 1.).reshape(-1, 28*28)
testX = (t.tensor(trainX, dtype=t.float) / 127.5 - 1.).reshape(-1, 28*28)
trainY = t.tensor(trainY, dtype=t.long)
testY = t.tensor(testY, dtype=t.long)
plt.imshow(trainX[0].reshape(28, 28), cmap='gray')
#plt.show()

def batch(batch = 128):
    indices = t.randperm(trainX.shape[0])[:batch]
    return trainX[indices], trainY[indices]

"""set, coor = batch(10)
image = set[0].reshape(28, 28)
plt.imshow(image, cmap="gray")
plt.show()"""

mlp = Mlp()
optimiseur = t.optim.Adam(mlp.parameters())
epochs = 500
losses = []
for epoch in tqdm(range(epochs)):
    x, label = batch()
    prediction = mlp(x)
    loss = nn.functional.nll_loss(prediction, label)
    optimiseur.zero_grad()
    loss.backward()
    optimiseur.step()
    losses.append(loss)

plt.plot(losses)
plt.show()