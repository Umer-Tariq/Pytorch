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


model = LinearRegression()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
epochs = 1000

loss_list = []

for epoch in range(epochs):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())


plt.plot(size = (10, 7))
plt.plot(range(epochs), loss_list, c = 'red', label = "Loss Curve")
plt.legend()
plt.show()

# Saving the model

torch.save(model.state_dict(), "model.pth")