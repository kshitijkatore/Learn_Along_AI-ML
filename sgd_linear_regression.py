import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class SGDLinearRegression:
    """
    Stochastic Gradient Descent Linear Regression implementation from scratch.
    Supports pure SGD, mini-batch SGD, and batch gradient descent.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        batch_size: int = 1,
        shuffle: bool = True,
        verbose: bool = False,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize SGD Linear Regression model.

        Parameters:
        - learning_rate: Step size for gradient updates
        - epochs: Number of passes through the entire dataset
        - batch_size: Number of samples per gradient update
                    (1 = pure SGD, >1 = mini-batch, n_samples = batch)
        - shuffle: Whether to shuffle data each epoch
        - verbose: Print training progress
        - random_seed: Seed for reproducibility
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_seed = random_seed

        self.weights = None
        self.bias = None
        self.cost_history = []
        self.iterations = 0

        if random_seed is not None:
            np.random.seed(random_seed)

    def _initialize_parameters(self, n_features: int) -> None:
        """Initialize weights and bias randomly."""
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.cost_history = []
        self.iterations = 0

    def _generate_batches(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate batches for training.

        Returns list of (X_batch, y_batch) tuples.
        """
        n_samples = X.shape[0]

        if self.shuffle:
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

        batches = []

        if self.batch_size >= n_samples:
            # Batch gradient descent
            batches.append((X, y))
        else:
            # Mini-batch or stochastic gradient descent
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i : i + self.batch_size]
                y_batch = y[i : i + self.batch_size]
                batches.append((X_batch, y_batch))

        return batches

    def _compute_gradients(
        self, X_batch: np.ndarray, y_batch: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for a batch.

        Returns:
        - dw: gradient for weights
        - db: gradient for bias
        """
        n = X_batch.shape[0]

        # Predictions
        y_pred = np.dot(X_batch, self.weights) + self.bias

        # Error
        error = y_pred - y_batch

        # Gradients
        dw = (2 / n) * np.dot(X_batch.T, error)
        db = (2 / n) * np.sum(error)

        return dw, db

    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Squared Error cost.
        """
        n = X.shape[0]
        y_pred = np.dot(X, self.weights) + self.bias
        mse = np.mean((y - y_pred) ** 2)
        return mse

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGDLinearRegression":
        """
        Train the model using SGD.

        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - y: Target vector (n_samples,)

        Returns:
        - self: Trained model
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Initialize parameters
        self._initialize_parameters(n_features)

        # Training loop
        for epoch in range(self.epochs):
            batches = self._generate_batches(X, y)

            epoch_cost = 0

            for X_batch, y_batch in batches:
                # Compute gradients
                dw, db = self._compute_gradients(X_batch, y_batch)

                # Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

                # Track cost
                batch_cost = self._compute_cost(X_batch, y_batch)
                epoch_cost += batch_cost * X_batch.shape[0]

                self.iterations += 1

            # Average cost for the epoch
            epoch_cost /= n_samples
            self.cost_history.append(epoch_cost)

            # Print progress
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - Cost: {epoch_cost:.4f}")

        if self.verbose:
            print(f"Training completed! Final cost: {self.cost_history[-1]:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters:
        - X: Feature matrix (n_samples, n_features)

        Returns:
        - predictions: Predicted values
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        return np.dot(X, self.weights) + self.bias

    def get_parameters(self) -> Tuple[np.ndarray, float]:
        """
        Get learned parameters.

        Returns:
        - weights: Learned weights
        - bias: Learned bias
        """
        return self.weights, self.bias

    def get_cost_history(self) -> List[float]:
        """
        Get cost history for analysis.

        Returns:
        - cost_history: List of costs per epoch
        """
        return self.cost_history

    def plot_cost_history(self, title: str = "Cost History") -> None:
        """
        Plot cost history over iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history, label="Training Cost")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


# Convenience functions for comparison


def batch_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 1000,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Batch Gradient Descent implementation for comparison.
    """
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features) * 0.01
    bias = 0.0
    cost_history = []

    for epoch in range(epochs):
        # Predictions
        y_pred = np.dot(X, weights) + bias
        error = y_pred - y

        # Gradients
        dw = (2 / n_samples) * np.dot(X.T, error)
        db = (2 / n_samples) * np.sum(error)

        # Update
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Cost
        cost = np.mean(error**2)
        cost_history.append(cost)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Cost: {cost:.4f}")

    return weights, bias, cost_history


def analytical_solution(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute analytical solution using normal equation.
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    bias = theta[0]
    weights = theta[1:]
    return weights, bias


# Utility functions for testing


def generate_synthetic_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise_level: float = 0.1,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic linear regression data.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features) * 2
    true_bias = np.random.randn() * 2

    y = np.dot(X, true_weights) + true_bias
    y += noise_level * np.random.randn(n_samples)

    return X, y


def compare_algorithms(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 1000,
    batch_sizes: List[int] = [1, 10, 50, 100],
    plot: bool = True,
) -> dict:
    """
    Compare different gradient descent variants.
    """
    results = {}

    # Analytical solution
    true_weights, true_bias = analytical_solution(X, y)
    true_cost = np.mean((y - (np.dot(X, true_weights) + true_bias)) ** 2)

    print(
        f"Analytical Solution - Weights: {true_weights}, Bias: {true_bias}, Cost: {true_cost:.4f}"
    )

    # Batch Gradient Descent
    batch_weights, batch_bias, batch_cost = batch_gradient_descent(
        X, y, learning_rate, epochs, verbose=False
    )
    batch_cost = batch_cost[-1]
    print(
        f"Batch GD - Weights: {batch_weights}, Bias: {batch_bias}, Cost: {batch_cost:.4f}"
    )
    results["batch"] = {
        "weights": batch_weights,
        "bias": batch_bias,
        "cost": batch_cost,
        "cost_history": batch_cost,
    }

    # SGD and Mini-batch variants
    for batch_size in batch_sizes:
        sgd = SGDLinearRegression(
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False,
        )

        sgd.fit(X, y)
        weights, bias = sgd.get_parameters()
        cost = sgd.cost_history[-1]

        print(
            f"SGD (batch_size={batch_size}) - Weights: {weights}, Bias: {bias}, Cost: {cost:.4f}"
        )

        results[f"sgd_{batch_size}"] = {
            "weights": weights,
            "bias": bias,
            "cost": cost,
            "cost_history": sgd.cost_history,
        }

    if plot:
        plt.figure(figsize=(12, 8))

        # Plot analytical solution
        plt.axhline(
            y=true_cost,
            color="black",
            linestyle="--",
            label=f"Analytical (Cost: {true_cost:.4f})",
        )

        # Plot batch GD
        plt.plot(
            batch_cost,
            label=f"Batch GD (Cost: {batch_cost:.4f})",
            color="blue",
            linewidth=2,
        )

        # Plot SGD variants
        for batch_size, result in results.items():
            if batch_size.startswith("sgd"):
                bs = int(batch_size.split("_")[1])
                plt.plot(
                    result["cost_history"],
                    label=f"SGD (bs={bs})",
                    alpha=0.7,
                    linewidth=1.5,
                )

        plt.xlabel("Epoch")
        plt.ylabel("Cost (MSE)")
        plt.title("Gradient Descent Variants Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()

    return results
