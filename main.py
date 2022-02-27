import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#0 Prepare data

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype.float32)
y = torch.from_numpy(y_numpy.astype.float32)
y = y.view(y.shape[0], 1)
n_samples, n_features = X.shape

#1 Model

input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

#2 Loss & optimizer



#3 Training loop

