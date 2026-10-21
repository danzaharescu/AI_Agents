"""A small autoregressive time-series forecasting example for learning."""

import matplotlib.pyplot as plt
import numpy as np


def make_monthly_series(length: int = 240, seed: int = 7) -> np.ndarray:
    """Create a monthly series with trend, seasonality, and autocorrelated noise."""
    rng = np.random.default_rng(seed)
    time = np.arange(length)

    # Combine three common time-series components:
    # - trend: the average level rises slowly over time
    # - seasonality: the pattern repeats every 12 months
    # - noise: unpredictable variation around the pattern
    trend = 0.03 * time
    yearly_seasonality = 2.5 * np.sin(2 * np.pi * time / 12)

    # Innovations are new random shocks. The recurrence below carries some of
    # each month's noise into the next month, creating autocorrelation.
    innovations = rng.normal(0, 0.65, length)
    noise = np.zeros(length)
    for index in range(1, length):
        noise[index] = 0.55 * noise[index - 1] + innovations[index]

    # A fixed seed makes the synthetic data reproducible between runs.
    return trend + yearly_seasonality + noise


def make_lagged_data(
    series: np.ndarray, n_lags: int
) -> tuple[np.ndarray, np.ndarray]:
    """Make supervised-learning examples from a one-dimensional time series.

    For example, with n_lags=3, the input for time t is
    [series[t-1], series[t-2], series[t-3]], and its target is series[t].
    """
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    if series.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if len(series) <= n_lags:
        raise ValueError("series must contain more values than n_lags")

    # Each row uses only observations that occurred before its target. The
    # nearest previous observation is first, followed by older observations.
    features = np.array(
        [
            [series[index - lag] for lag in range(1, n_lags + 1)]
            for index in range(n_lags, len(series))
        ]
    )
    targets = series[n_lags:]
    return features, targets


def fit_autoregression(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Fit a linear autoregression by least squares, including an intercept.

    The model learns target ~= intercept + weight_1 * lag_1 + ... .
    Least squares chooses weights that minimize the sum of squared training
    prediction errors without requiring an iterative training loop.
    """
    # A column of ones lets least squares learn the model's intercept (bias).
    design_matrix = np.column_stack((np.ones(len(features)), features))
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, targets, rcond=None)
    return coefficients


def forecast_autoregression(
    history: np.ndarray, coefficients: np.ndarray, steps: int
) -> np.ndarray:
    """Forecast multiple steps ahead by feeding each prediction back as a lag.

    At the first step, all lags come from observed history. At later steps,
    older predictions become inputs because future actual values are unknown.
    """
    n_lags = len(coefficients) - 1
    if n_lags < 1:
        raise ValueError("coefficients must include an intercept and at least one lag")
    if len(history) < n_lags:
        raise ValueError("history must contain at least n_lags values")
    if steps < 0:
        raise ValueError("steps cannot be negative")

    extended_history = list(history)
    predictions = []
    for _ in range(steps):
        # Keep the lag order consistent with make_lagged_data: most recent first.
        recent_values = np.array(
            [extended_history[-lag] for lag in range(1, n_lags + 1)]
        )
        # coefficients[0] is the intercept; the remaining coefficients match
        # the lag values in recent_values.
        next_value = coefficients[0] + np.dot(coefficients[1:], recent_values)
        predictions.append(next_value)
        # Appending the estimate allows it to be used as an input in the next
        # forecast step, rather than leaking the test-set actual value.
        extended_history.append(float(next_value))

    return np.array(predictions)


def main() -> None:
    # Keep the final 20% for testing, in chronological order. Shuffling would
    # let the model learn from future observations.
    series = make_monthly_series()
    split_index = int(len(series) * 0.8)
    train_series = series[:split_index]
    test_series = series[split_index:]

    n_lags = 12  # Use the previous year of monthly values as model inputs.
    # Fit only on the training period. The first n_lags observations provide
    # context, so they are not targets in the lagged training examples.
    features, targets = make_lagged_data(train_series, n_lags)
    coefficients = fit_autoregression(features, targets)
    # Forecast the whole test window recursively from the end of training.
    predictions = forecast_autoregression(
        train_series, coefficients, len(test_series)
    )

    # Compare the model with a simple baseline that repeats the final training
    # observation for every future month.
    baseline = np.full(len(test_series), train_series[-1])
    # RMSE (root mean squared error) measures typical error in the series'
    # original units; smaller values are better.
    model_rmse = np.sqrt(np.mean((test_series - predictions) ** 2))
    baseline_rmse = np.sqrt(np.mean((test_series - baseline) ** 2))

    print(f"Training observations: {len(train_series)}")
    print(f"Test observations:     {len(test_series)}")
    print(f"Autoregression RMSE:   {model_rmse:.3f}")
    print(f"Persistence RMSE:      {baseline_rmse:.3f}")

    # Plot the test forecast at its original time positions, after the split.
    time = np.arange(len(series))
    test_time = time[split_index:]
    plt.figure(figsize=(12, 5))
    plt.plot(time[:split_index], train_series, label="Training data")
    plt.plot(test_time, test_series, label="Actual test data", color="black")
    plt.plot(test_time, predictions, label="AR(12) forecast", color="tab:red")
    plt.axvline(split_index, color="gray", linestyle="--", label="Train/test split")
    plt.title("Monthly Time Series: Recursive Autoregressive Forecast")
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
