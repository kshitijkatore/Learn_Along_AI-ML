"""
Example usage and demonstration of Stochastic Gradient Descent implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sgd_linear_regression import (
    SGDLinearRegression,
    generate_synthetic_data,
    compare_algorithms,
)

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
print("Generating synthetic data...")
X, y = generate_synthetic_data(
    n_samples=1000, n_features=1, noise_level=0.5, random_seed=42
)

# Test pure SGD
print("\nTesting pure Stochastic Gradient Descent...")
sgd_pure = SGDLinearRegression(
    learning_rate=0.01,
    epochs=500,
    batch_size=1,  # Pure SGD
    shuffle=True,
    verbose=True,
    random_seed=42,
)

sgd_pure.fit(X, y)
weights, bias = sgd_pure.get_parameters()
print(f"Pure SGD - Final Weights: {weights}, Bias: {bias}")
sgd_pure.plot_cost_history("Pure Stochastic Gradient Descent Cost History")

# Test mini-batch SGD
print("\nTesting Mini-batch Stochastic Gradient Descent...")
sgd_mini = SGDLinearRegression(
    learning_rate=0.01,
    epochs=500,
    batch_size=32,  # Mini-batch
    shuffle=True,
    verbose=True,
    random_seed=42,
)

sgd_mini.fit(X, y)
weights, bias = sgd_mini.get_parameters()
print(f"Mini-batch SGD - Final Weights: {weights}, Bias: {bias}")
sgd_mini.plot_cost_history("Mini-batch Stochastic Gradient Descent Cost History")

# Test batch gradient descent
print("\nTesting Batch Gradient Descent...")
sgd_batch = SGDLinearRegression(
    learning_rate=0.01,
    epochs=500,
    batch_size=X.shape[0],  # Batch (all samples)
    shuffle=True,
    verbose=True,
    random_seed=42,
)

sgd_batch.fit(X, y)
weights, bias = sgd_batch.get_parameters()
print(f"Batch GD - Final Weights: {weights}, Bias: {bias}")
sgd_batch.plot_cost_history("Batch Gradient Descent Cost History")

# Compare all algorithms
print("\nComparing all algorithms...")
results = compare_algorithms(
    X,
    y,
    learning_rate=0.01,
    epochs=500,
    batch_sizes=[1, 32, 64, 128, X.shape[0]],
    plot=True,
)

# Make predictions and visualize
print("\nMaking predictions and visualizing results...")

# Get predictions from different models
y_pred_pure = sgd_pure.predict(X)
y_pred_mini = sgd_mini.predict(X)
y_pred_batch = sgd_batch.predict(X)

# Plot data and predictions
plt.figure(figsize=(12, 8))

# Scatter plot of actual data
plt.scatter(X, y, alpha=0.5, label="Actual Data", color="gray")

# Plot predictions
plt.plot(
    X, y_pred_pure, label=f"Pure SGD (w={weights[0]:.2f})", color="red", linewidth=2
)
plt.plot(
    X,
    y_pred_mini,
    label=f"Mini-batch SGD (w={weights[0]:.2f})",
    color="green",
    linewidth=2,
)
plt.plot(
    X, y_pred_batch, label=f"Batch GD (w={weights[0]:.2f})", color="blue", linewidth=2
)

plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.title("SGD vs Batch Gradient Descent Comparison")
plt.legend()
plt.grid(True)
plt.show()

# Print final comparison summary
print("\n=== Final Comparison Summary ===")
print(f"Pure SGD (batch_size=1) - Final Cost: {sgd_pure.cost_history[-1]:.4f}")
print(f"Mini-batch SGD (batch_size=32) - Final Cost: {sgd_mini.cost_history[-1]:.4f}")
print(
    f"Batch GD (batch_size={X.shape[0]}) - Final Cost: {sgd_batch.cost_history[-1]:.4f}"
)

# Additional analysis
print("\n=== Learning Rate Sensitivity ===")
learning_rates = [0.001, 0.01, 0.1, 0.5]
for lr in learning_rates:
    sgd_test = SGDLinearRegression(
        learning_rate=lr, epochs=200, batch_size=32, verbose=False
    )
    sgd_test.fit(X, y)
    print(f"Learning Rate {lr}: Final Cost = {sgd_test.cost_history[-1]:.4f}")

print("\n=== Batch Size Sensitivity ===")
for bs in [1, 8, 32, 64, 128, 256, X.shape[0]]:
    sgd_test = SGDLinearRegression(
        learning_rate=0.01, epochs=200, batch_size=bs, verbose=False
    )
    sgd_test.fit(X, y)
    print(f"Batch Size {bs}: Final Cost = {sgd_test.cost_history[-1]:.4f}")

print("\n=== Demonstration Complete! ===")
print("The implementation successfully demonstrates:")
print("- Pure Stochastic Gradient Descent")
print("- Mini-batch Stochastic Gradient Descent")
print("- Batch Gradient Descent")
print("- Learning rate and batch size sensitivity")
print("- Cost convergence visualization")
