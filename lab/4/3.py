import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Define the Deep Feed Forward ANN model
class DeepANN(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, activation_func):
        super(DeepANN, self).__init__()
        self.activation_func = activation_func
        layers = []
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                if activation_func == 'relu':
                    layers.append(nn.ReLU())
                elif activation_func == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation_func == 'tanh':
                    layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Load data
data = load_iris()
X = data.data
y = (data.target == 0).astype(int)  # Binary classification task
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Use only the tanh activation function
activation_function = 'tanh'
results = []

model = DeepANN(input_size=4, hidden_layer_sizes=[10, 20, 30, 40, 30, 20, 10, 10, 5, 5], output_size=1, activation_func=activation_function)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(torch.sigmoid(outputs), y_train)
    loss.backward()
    optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    accuracy = accuracy_score(y_test, predicted)
    results.append((activation_function, accuracy))

# Display results
print("Activation Function Performance:")
print("Activation Function | Accuracy")
print("--------------------|---------")
for func, accuracy in results:
    print(f"{func:<18} | {accuracy:.4f}")





"""
use the best activation function found in problem 2
and then when the model is deeper, model perform better with less training epochs!

"""