# Compare Neural Tangent Kernel (NTK) and Neural Network (NN) on a synthetic dataset

#%%
# generate data
import numpy as np

def generate_data(n, d=10, m=5, seed=42):
    np.random.seed(seed)
    
    # Generate n points uniformly on the unit sphere in d dimensions
    x = np.random.randn(n, d)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    # Generate labels
    y = np.zeros(n)
    for i in range(n):
        for j in range(m):
            e_j = np.zeros(d)
            e_j[j] = 1
            y[i] += np.maximum(0, x[i,j])
        y[i] /= m
    
    return x, y
# %%
import torch
from sklearn.kernel_ridge import KernelRidge
import torch.nn as nn
import torch.optim as optim

# Kernel regression function
def ntk(X, Y):
    dot_product = np.dot(X, Y.T)
    dot_product = np.clip(dot_product, -1, 1)
    return (dot_product * (np.pi - np.arccos(dot_product))) / (2 * np.pi)

def kernel_regression(X_train, y_train):
    model = KernelRidge(alpha=1, kernel=ntk)
    model.fit(X_train, y_train)
    return model

# Two-layer neural network
class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TwoLayerNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        hidden = self.activation(self.fc1(x))
        output = self.fc2(hidden)
        return output.squeeze()

# Train neural network
def train_nn(model, X_train, y_train, lr=0.01, epochs=1000):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

#%%
# Generate datasets for different values of n
datasets = {}
for n in [20, 40, 80, 160]:
    datasets[n] = generate_data(n)

datasets[20][0].shape

# Verify that rows of x have norm 1
for n, (X, y) in datasets.items():
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1), f"Not all rows have norm 1 for n={n}"

#%%
# Evaluate models
for n, (X, y) in datasets.items():
    # Kernel regression
    model = kernel_regression(X, y)
    y_pred_kernel = model.predict(X)
    kernel_mse = np.mean((y_pred_kernel - y) ** 2)
    print(f"Kernel regression MSE for n={n}: {kernel_mse}")

    # Neural network
    input_dim = X.shape[1]
    hidden_dim = 50  # You can tune this parameter
    model = TwoLayerNN(input_dim, hidden_dim)
    train_nn(model, X, y)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_pred_nn = model(X_tensor).numpy()
    nn_mse = np.mean((y_pred_nn - y) ** 2)
    print(f"Neural network MSE for n={n}: {nn_mse}")
    
# %%
import matplotlib.pyplot as plt

# Generate test data
X_test, y_test = generate_data(1000)

# Store test errors
test_errors_kernel = []
test_errors_nn = []

# Evaluate models on test data
for n, (X_train, y_train) in datasets.items():
    # Kernel regression
    model = kernel_regression(X_train, y_train)
    y_pred_kernel = model.predict(X_test)
    kernel_mse = np.mean((y_pred_kernel - y_test) ** 2)
    test_errors_kernel.append(kernel_mse)

    # Neural network
    input_dim = X_train.shape[1]
    hidden_dim = 50  # You can tune this parameter
    model = TwoLayerNN(input_dim, hidden_dim)
    train_nn(model, X_train, y_train)
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred_nn = model(X_test_tensor).numpy()
    nn_mse = np.mean((y_pred_nn - y_test) ** 2)
    test_errors_nn.append(nn_mse)

# Plot test errors
plt.figure(figsize=(10, 6))
plt.plot(datasets.keys(), test_errors_kernel, label='Neural Tangent Kernel Regression', marker='o')
plt.plot(datasets.keys(), test_errors_nn, label='Neural Network, n_hidden=50', marker='o')
plt.xlabel('Number of Training Data')
plt.ylabel('Test Error (MSE)')
plt.title('Test Error vs Number of Training Data')
plt.legend()
plt.grid(True)
plt.show()