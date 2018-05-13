import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from rnn import RNN
import numpy as np
from torch import nn
import torch
from torch.autograd import Variable


ss_X_dep = StandardScaler()
ss_y_dep = StandardScaler()

def rmse(y1, y2):
    return np.sqrt(mean_squared_error(y1, y2))

# Noted that the demo data are processed manually, so they are not real data,
# but they still can reflect the correlation between the original data.
data = pd.read_csv('data/demo.csv')

Inputs = data.drop('Year', axis=1).drop('Depth', axis=1)
Outputs = data['Depth']

Inputs = Inputs.as_matrix()
Outputs = Outputs.as_matrix().reshape(-1, 1)

# First 12 years of data
X_train_dep = Inputs[0:144]
y_train_dep = Outputs[0:144]

# Last 2 years of data
X_test_dep = Inputs[144:]

print("X_train_dep shape", X_train_dep.shape)
print("y_train_dep shape", y_train_dep.shape)
print("X_test_dep shape", X_test_dep.shape)

X = np.concatenate([X_train_dep, X_test_dep], axis=0)

# Standardization
X = ss_X_dep.fit_transform(X)

# First 12 years of data
X_train_dep_std = X[0:144]
y_train_dep_std = ss_y_dep.fit_transform(y_train_dep)

# All 14 years of data
X_test_dep_std  = X
X_train_dep_std = np.expand_dims(X_train_dep_std, axis=0)
y_train_dep_std = np.expand_dims(y_train_dep_std, axis=0)

X_train_dep_std = Variable(torch.from_numpy(X_train_dep_std).float())
y_train_dep_std = Variable(torch.from_numpy(y_train_dep_std).float())


model = RNN(input_size=5, hidden_size=40, num_layers=1, class_size=1, dropout=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)   # optimize all cnn parameters
loss_func = nn.MSELoss()

for i in range(20):
    prediction = model(X_train_dep_std)
    loss = loss_func(prediction, y_train_dep_std) # cross entropy loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()
    print(loss)

