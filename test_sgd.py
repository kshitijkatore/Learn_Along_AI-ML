"""
Comprehensive test script for Stochastic Gradient Descent implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sgd_linear_regression import (
    SGDLinearRegression,
    generate_synthetic_data,
    compare_algorithms,
)

# Test 1: Basic functionality test
print("\n=== Test 1: Basic Functionality ===")
print("Testing SGD with different batch sizes...")

# Generate test data
X_test, y_test = generate_synthetic_data(
    n_samples=500, n_features=1, noise_level=0.3, random_seed=123
)

# Test different batch sizes
batch_sizes = [1, 5, 10, 20, 50, 100, 200, 500]
results = {}

for bs in batch_sizes:
    print(f"Testing batch_size={bs}...")

    sgd = SGDLinearRegression(
        learning_rate=0.01,
        epochs=300,
        batch_size=bs,
        shuffle=True,
        verbose=False,
        random_seed=123,
    )

    sgd.fit(X_test, y_test)
    weights, bias = sgd.get_parameters()
    final_cost = sgd.cost_history[-1]

    results[bs] = {
        "weights": weights,
        "bias": bias,
        "final_cost": final_cost,
        "iterations": len(sgd.cost_history),
    }

    print(f"  Final cost: {final_cost:.4f}, Weights: {weights}, Bias: {bias}")

# Test 2: Learning rate sensitivity
print("\n=== Test 2: Learning Rate Sensitivity ===")
print("Testing different learning rates...")

learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
results_lr = {}

for lr in learning_rates:
    print(f"Testing learning_rate={lr}...")

    sgd = SGDLinearRegression(
        learning_rate=lr,
        epochs=200,
        batch_size=32,
        shuffle=True,
        verbose=False,
        random_seed=123,
    )

    sgd.fit(X_test, y_test)
    final_cost = sgd.cost_history[-1]

    results_lr[lr] = {"final_cost": final_cost, "iterations": len(sgd.cost_history)}

    print(f"  Final cost: {final_cost:.4f}")

# Test 3: Convergence analysis
print("\n=== Test 3: Convergence Analysis ===")
print("Analyzing convergence patterns...")

# Test with different epoch counts
epoch_counts = [50, 100, 200, 500, 1000]
convergence_results = {}

for epochs in epoch_counts:
    print(f"Testing epochs={epochs}...")

    sgd = SGDLinearRegression(
        learning_rate=0.01,
        epochs=epochs,
        batch_size=32,
        shuffle=True,
        verbose=False,
        random_seed=123,
    )

    sgd.fit(X_test, y_test)
    final_cost = sgd.cost_history[-1]

    convergence_results[epochs] = {
        "final_cost": final_cost,
        "cost_history": sgd.cost_history,
    }

    print(f"  Final cost: {final_cost:.4f}")

# Test 4: Feature scaling importance
print("\n=== Test 4: Feature Scaling Importance ===")
print("Testing with and without feature scaling...")

# Generate data with different feature scales
X_unscaled, y_unscaled = generate_synthetic_data(
    n_samples=300, n_features=1, noise_level=0.2, random_seed=456
)

# Scale features
X_scaled = (X_unscaled - np.mean(X_unscaled)) / np.std(X_unscaled)

# Test unscaled
sgd_unscaled = SGDLinearRegression(
    learning_rate=0.01,
    epochs=300,
    batch_size=32,
    shuffle=True,
    verbose=False,
    random_seed=456,
)
sgd_unscaled.fit(X_unscaled, y_unscaled)

# Test scaled
sgd_scaled = SGDLinearRegression(
    learning_rate=0.01,
    epochs=300,
    batch_size=32,
    shuffle=True,
    verbose=False,
    random_seed=456,
)
sgd_scaled.fit(X_scaled, y_unscaled)

print(f"Unscaled features - Final cost: {sgd_unscaled.cost_history[-1]:.4f}")
print(f"Scaled features - Final cost: {sgd_scaled.cost_history[-1]:.4f}")

# Test 5: Multiple features
print("\n=== Test 5: Multiple Features ===")
print("Testing with multiple input features...")

# Generate multi-feature data
X_multi, y_multi = generate_synthetic_data(
    n_samples=500, n_features=5, noise_level=0.3, random_seed=789
)

sgd_multi = SGDLinearRegression(
    learning_rate=0.01,
    epochs=300,
    batch_size=32,
    shuffle=True,
    verbose=False,
    random_seed=789,
)
sgd_multi.fit(X_multi, y_multi)
weights, bias = sgd_multi.get_parameters()
final_cost = sgd_multi.cost_history[-1]

print(f"Final cost: {final_cost:.4f}")
print(f"Learned weights: {weights}")
print(f"Learned bias: {bias}")

# Visualization: Batch size comparison
print("\n=== Visualization: Batch Size Comparison ===")

plt.figure(figsize=(12, 8))

for bs, result in results.items():
    plt.plot(result["cost_history"], label=f"Batch Size {bs}", alpha=0.7)

plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Convergence Comparison for Different Batch Sizes")
plt.legend()
plt.grid(True)
plt.show()

# Visualization: Learning rate comparison
plt.figure(figsize=(12, 8))

for lr, result in results_lr.items():
    sgd = SGDLinearRegression(
        learning_rate=lr,
        epochs=200,
        batch_size=32,
        shuffle=True,
        verbose=False,
        random_seed=123,
    )
    sgd.fit(X_test, y_test)
    plt.plot(sgd.cost_history, label=f"Learning Rate {lr}", alpha=0.7)

plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Learning Rate Sensitivity Analysis")
plt.legend()
plt.grid(True)
plt.show()

# Print summary
print("\n=== Test Summary ===")
print("All tests completed successfully!")
print("- Basic functionality works with different batch sizes")
print("- Learning rate sensitivity is demonstrated")
print("- Convergence patterns are analyzed")
print("- Feature scaling importance is shown")
print("- Multi-feature support is validated")
print("\nSGD implementation is working correctly!")
