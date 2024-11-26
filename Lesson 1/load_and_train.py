import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        return x
    
weight = 0.7
bias = 0.3

X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = weight * X + bias

indices = torch.randperm(len(X))
train_size = int(0.8 * len(X))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

# Set the manual seed before making the model so that the parameters are initialized as same

torch.manual_seed(42)

model = LinearRegression()

model.load_state_dict(torch.load("C:\Coding Tutorials\Pytorch\model.pth"))

model.eval()

with torch.inference_mode():
    preds = model(X_test)

plt.plot(size = (10, 7))
plt.plot(X, y, c = 'red', label = 'Original')
plt.plot(X_test, preds, c = 'green', label = 'Predicted')
plt.legend()
plt.show()

# Check the device of the model, parameters and the data:
model.device

# Model.parameters return a iterable containing the parameters of the model. Next returns the first of these. It is used to traverse the iterable one by one. If the first parameter is on the GPU, then all are on GPU

next(model.parameters()).device
data.device