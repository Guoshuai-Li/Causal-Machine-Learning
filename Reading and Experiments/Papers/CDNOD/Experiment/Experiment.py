import numpy as np
import matplotlib.pyplot as plt
from causallearn.search.ConstraintBased.CDNOD import cdnod

np.random.seed(42)

def generate_data(n_samples=2000):
    """
    Generate 2-variable time series with independent mechanism changes:
    
    Regime 1 (t=0-666):
        X(t) = 0.3*X(t-1) + noise    [X mechanism: weak AR]
        Y(t) = 0.5*Y(t-1) + 0.6*X(t-1) + noise
    
    Regime 2 (t=667-1333):
        X(t) = 0.7*X(t-1) + noise    [X mechanism: strong AR]
        Y(t) = 0.5*Y(t-1) + 0.6*X(t-1) + noise  [Y unchanged]
    
    Regime 3 (t=1334-2000):
        X(t) = 0.3*X(t-1) + noise    [X mechanism: back to weak]
        Y(t) = 0.5*Y(t-1) + 0.6*X(t-1) + noise  [Y unchanged]
    
    Key insight: X's mechanism changes, Y's mechanism stays constant
    This should help CDNOD identify X --> Y
    """
    data = np.zeros((n_samples, 2))
    regime_indicator = np.zeros((n_samples, 1))
    regime_changes = [0, 667, 1334, 2000]
    x_ar_coeffs = [0.3, 0.7, 0.3]  # X's AR coefficient changes
    
    for t in range(1, n_samples):
        # Determine current regime
        regime_idx = 0
        for i in range(len(regime_changes)-1):
            if regime_changes[i] <= t < regime_changes[i+1]:
                regime_idx = i
                break
        
        x_ar_coef = x_ar_coeffs[regime_idx]
        regime_indicator[t] = regime_idx
        
        # X: mechanism changes across regimes
        data[t, 0] = x_ar_coef * data[t-1, 0] + np.random.randn() * 0.5
        
        # Y: mechanism stays constant (always depends on X with coef=0.6)
        data[t, 1] = 0.5 * data[t-1, 1] + 0.6 * data[t-1, 0] + np.random.randn() * 0.5
    
    return data, regime_indicator, regime_changes

def visualize_data(data, regime_changes):
    """Visualize time series with regime boundaries"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    var_names = ['X', 'Y']
    colors = ['blue', 'orange']
    
    for i, (ax, name, color) in enumerate(zip(axes, var_names, colors)):
        ax.plot(data[:, i], linewidth=0.7, color=color, alpha=0.8)
        
        # Add regime boundaries
        for change_point in regime_changes[1:-1]:
            ax.axvline(x=change_point, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_ylabel(name, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, len(data))
    
    axes[0].set_title('Nonstationary Time Series (Independent Mechanism Changes)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time', fontsize=12)
    
    # Add regime labels showing X's AR coefficient
    regime_labels = ['R1\n(X:0.3)', 'R2\n(X:0.7)', 'R3\n(X:0.3)']
    for i, label in enumerate(regime_labels):
        mid_point = (regime_changes[i] + regime_changes[i+1]) / 2
        axes[0].text(mid_point, axes[0].get_ylim()[1]*0.85, label, 
                    ha='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def run_cdnod(data, regime_indicator):
    """
    Run CDNOD algorithm
    
    Args:
        data: (n_samples, 2) array [X, Y]
        regime_indicator: (n_samples, 1) array
    
    Returns:
        CausalGraph object
    """
    cg = cdnod(data, regime_indicator, alpha=0.05)
    return cg

def print_results(cg):
    """Print CDNOD results"""

    print("CDNOD RESULT")
    print("\nCausal Graph:")
    print(cg.G)
    print("\nExperimental Design:")
    print("- X's mechanism changes: AR coef = 0.3 -> 0.7 -> 0.3")
    print("- Y's mechanism constant: always depends on X (coef=0.6)")
    print("\nExpected: X --> Y")
    print("(Because X changes independently, Y depends on X)")

if __name__ == "__main__":
    print("CDNOD: Independent Change Principle Experiment")
    print("Design: X's mechanism changes, Y's stays constant")
    print("Expected: X --> Y\n")
    
    # Step 1: Generate data
    data, regime_indicator, regime_changes = generate_data(n_samples=2000)
    visualize_data(data, regime_changes)
    
    # Step 2: Run CDNOD
    cg = run_cdnod(data, regime_indicator)
    
    # Step 3: Print results
    print_results(cg)
