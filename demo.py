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
X_test_dep_std = np.expand_dims(X_test_dep_std, axis=0)

# Transfer to Pytorch Variable
X_train_dep_std = Variable(torch.from_numpy(X_train_dep_std).float())
y_train_dep_std = Variable(torch.from_numpy(y_train_dep_std).float())
X_test_dep_std = Variable(torch.from_numpy(X_test_dep_std).float())

# Define rnn model
# You can also choose rnn_type as 'rnn' or 'gru'
model = RNN(input_size=5, hidden_size=40, num_layers=1, class_size=1, dropout=0.5, rnn_type='lstm')
# Define optimization function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)   # optimize all rnn parameters
# Define loss function
loss_func = nn.MSELoss()

# Start training
for iter in range(20000+1):
    model.train()
    prediction = model(X_train_dep_std)
    loss = loss_func(prediction, y_train_dep_std)
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()        # back propagation, compute gradients
    optimizer.step()
    if iter % 100 == 0:
        print("iteration: %s, loss: %s" % (iter, loss.item()))

# Save model
save_filename = 'checkpoints/LSTM_FC.pth'
torch.save(model, save_filename)
print('Saved as %s' % save_filename)

# Start evaluating model
model.eval()

y_pred_dep_ = model(X_test_dep_std).detach().numpy()
y_pred_dep = ss_y_dep.inverse_transform(y_pred_dep_[0, 144:])

print('the value of R-squared of Evaporation is ', r2_score(Outputs[144:], y_pred_dep))
print('the value of Root mean squared error of Evaporation is ', rmse(Outputs[144:], y_pred_dep))

f, ax1 = plt.subplots(1, 1, sharex=True, figsize=(6, 4))

ax1.plot(Outputs[144:], color="blue", linestyle="-", linewidth=1.5, label="Measurements")
ax1.plot(y_pred_dep, color="green", linestyle="--", linewidth=1.5, label="Proposed model")

plt.legend(loc='upper right')
plt.xticks(fontsize=8,fontweight='normal')
plt.yticks(fontsize=8,fontweight='normal')
plt.xlabel('Time (Month)', fontsize=10)
plt.ylabel('Water table depth (m)', fontsize=10)
plt.xlim(0, 25)
plt.savefig('results.png', format='png')
plt.show()


##### Loading Model #####
model = torch.load('checkpoints/LSTM_FC.pth')
model.eval()
y_pred_dep_ = model(X_test_dep_std).detach().numpy()
y_pred_dep = ss_y_dep.inverse_transform(y_pred_dep_[0, 144:])

print('the value of R-squared of Evaporation is ', r2_score(Outputs[144:], y_pred_dep))
print('the value of Root mean squared error of Evaporation is ', rmse(Outputs[144:], y_pred_dep))
