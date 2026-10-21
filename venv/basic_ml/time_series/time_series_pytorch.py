"""A PyTorch neural-network example for autoregressive forecasting."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def make_monthly_series(length: int = 240, seed: int = 7) -> np.ndarray:
    """Create monthly data with trend, yearly seasonality, and noisy variation."""
    rng = np.random.default_rng(seed)
    time = np.arange(length)

    # Synthetic data combines a slowly rising trend, a repeating yearly pattern,
    # and noise that carries over from one month to the next.
    trend = 0.03 * time
    yearly_seasonality = 2.5 * np.sin(2 * np.pi * time / 12)
    innovations = rng.normal(0, 0.65, length)
    noise = np.zeros(length)
    for index in range(1, length):
        noise[index] = 0.55 * noise[index - 1] + innovations[index]

    return trend + yearly_seasonality + noise


def make_lagged_data(
    series: np.ndarray, n_lags: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create input windows of past values and targets for the next value."""
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    if series.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if len(series) <= n_lags:
        raise ValueError("series must contain more values than n_lags")

    # The most recent prior value is the first feature in each row.
    features = np.array(
        [
            [series[index - lag] for lag in range(1, n_lags + 1)]
            for index in range(n_lags, len(series))
        ]
    )
    return features, series[n_lags:]


def train_model(
    features: torch.Tensor,
    targets: torch.Tensor,
    n_lags: int,
    epochs: int = 600,
) -> nn.Sequential:
    """Train a small neural network to predict the next value from previous lags."""
    # The input is the previous year's values. A hidden layer lets the network
    # learn non-linear relationships, unlike the linear AR model in the other demo.
    model = nn.Sequential(
        nn.Linear(n_lags, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    # MSE is appropriate for a numeric forecasting target. Adam updates the
    # network weights using gradients computed automatically by PyTorch.
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        predictions = model(features)
        loss = loss_function(predictions, targets)

        # Clear old gradients, calculate this batch's gradients, then update
        # the weights to reduce the prediction error.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch in (0, epochs - 1):
            print(f"Epoch {epoch + 1:>3}/{epochs} - training MSE: {loss.item():.4f}")

    return model


def forecast_recursively(
    model: nn.Module,
    history: np.ndarray,
    n_lags: int,
    steps: int,
) -> np.ndarray:
    """Forecast multiple steps, feeding each prediction into the next input."""
    extended_history = list(history)
    predictions = []

    # Evaluation mode disables training-only behavior; no_grad avoids building
    # gradient history because forecasting does not update model weights.
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            # Match the feature order used to train the model: most recent lag first.
            recent_values = [extended_history[-lag] for lag in range(1, n_lags + 1)]
            model_input = torch.tensor([recent_values], dtype=torch.float32)
            next_value = model(model_input).item()
            predictions.append(next_value)
            # Future actual values are unknown, so use this estimate for the
            # next step rather than reading from the held-out test data.
            extended_history.append(next_value)

    return np.array(predictions)


def main() -> None:
    # Seed both libraries so the generated data, initial weights, and training
    # results are repeatable for learning and debugging.
    torch.manual_seed(7)

    series = make_monthly_series(seed=7)
    split_index = int(len(series) * 0.8)
    train_series = series[:split_index]
    test_series = series[split_index:]
    n_lags = 12

    # Standardize using training data only. This keeps values near zero and
    # prevents information from the future test period leaking into training.
    training_mean = float(np.mean(train_series))
    training_std = float(np.std(train_series))
    if training_std == 0:
        raise ValueError("Training series must have non-zero standard deviation")
    normalized_train = (train_series - training_mean) / training_std

    # Convert each history window and its next value into PyTorch tensors.
    feature_array, target_array = make_lagged_data(normalized_train, n_lags)
    features = torch.tensor(feature_array, dtype=torch.float32)
    targets = torch.tensor(target_array, dtype=torch.float32).reshape(-1, 1)

    model = train_model(features, targets, n_lags)

    # Forecast in standardized units, then convert predictions back to the
    # original scale so metrics and plots are easy to interpret.
    normalized_predictions = forecast_recursively(
        model,
        normalized_train,
        n_lags,
        len(test_series),
    )
    predictions = normalized_predictions * training_std + training_mean

    # Persistence predicts that the last training value remains unchanged.
    baseline = np.full(len(test_series), train_series[-1])
    model_rmse = np.sqrt(np.mean((test_series - predictions) ** 2))
    baseline_rmse = np.sqrt(np.mean((test_series - baseline) ** 2))

    print(f"Training observations: {len(train_series)}")
    print(f"Test observations:     {len(test_series)}")
    print(f"PyTorch AR(12) RMSE:   {model_rmse:.3f}")
    print(f"Persistence RMSE:      {baseline_rmse:.3f}")

    # Show the training history, actual held-out values, and recursive forecast.
    time = np.arange(len(series))
    test_time = time[split_index:]
    plt.figure(figsize=(12, 5))
    plt.plot(time[:split_index], train_series, label="Training data")
    plt.plot(test_time, test_series, label="Actual test data", color="black")
    plt.plot(test_time, predictions, label="PyTorch AR(12) forecast", color="tab:red")
    plt.axvline(split_index, color="gray", linestyle="--", label="Train/test split")
    plt.title("PyTorch Time-Series Forecast")
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
