import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

np.random.seed(42)


def generate_data(n=500):
    """
    Generate unidirectional causality data: X → Y
    X(t) = 0.8*X(t-1) + ε₁
    Y(t) = 0.5*Y(t-1) + 0.6*X(t-1) + ε₂
    """
    X = np.zeros(n)
    Y = np.zeros(n)
    
    for t in range(1, n):
        X[t] = 0.8 * X[t-1] + np.random.normal()
        Y[t] = 0.5 * Y[t-1] + 0.6 * X[t-1] + np.random.normal()
    
    return pd.DataFrame({'X': X, 'Y': Y})


def test_stationarity(series, name):
    """ADF test for stationarity"""
    result = adfuller(series, autolag='AIC')
    print(f"{name}: ADF={result[0]:.3f}, p-value={result[1]:.4f}")


def test_granger(data, direction, maxlag=5):
    """Granger causality test"""
    test_data = data[[data.columns[1], data.columns[0]]]
    results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
    
    print(f"\n{direction}:")
    for lag in range(1, maxlag + 1):
        f_test = results[lag][0]['ssr_ftest']
        f_stat = f_test[0]
        p_value = f_test[1]
        sig = '***' if p_value < 0.01 else '**' if p_value < 0.05 else ''
        print(f"  Lag {lag}: F={f_stat:.2f}, p={p_value:.4f} {sig}")


def plot_data(data):
    """Plot time series"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].plot(data['X'])
    axes[0].set_ylabel('X(t)')
    axes[1].plot(data['Y'])
    axes[1].set_ylabel('Y(t)')
    axes[1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate data
    data = generate_data(n=500)
    
    # Stationarity tests
    print("=== Stationarity Tests ===")
    test_stationarity(data['X'], 'X')
    test_stationarity(data['Y'], 'Y')
    
    # Granger causality tests
    print("\n=== Granger Causality Tests ===")
    test_granger(data[['X', 'Y']], 'X → Y')
    test_granger(data[['Y', 'X']], 'Y → X')
    
    # Plot
    print("\n=== Visualization ===")
    plot_data(data)
