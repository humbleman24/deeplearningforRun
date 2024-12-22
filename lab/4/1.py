import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Generate a simple regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Locally Weighted Regression
def locally_weighted_regression(X_train, y_train, X_test, tau=1.0):
    m = X_train.shape[0]
    y_pred = []
    for x in X_test:
        weights = np.exp(-np.sum((X_train - x) ** 2, axis=1) / (2 * tau ** 2))  # Gaussian weights
        W = np.diag(weights)  # Construct weight matrix
        X_bias = np.hstack([np.ones((m, 1)), X_train])  # Add bias term
        theta = np.linalg.inv(X_bias.T @ W @ X_bias) @ X_bias.T @ W @ y_train  # Compute parameters
        y_pred.append(np.hstack([1, x]) @ theta)  # Predict
    return np.array(y_pred)

# Run LWR on the data
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = locally_weighted_regression(X, y, X_test, tau=1.0)

# Plot the results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_test, y_pred, color='red', label='LWR fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
