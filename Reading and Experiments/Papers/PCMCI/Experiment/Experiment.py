import numpy as np
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# Set random seed for reproducibility
np.random.seed(42)

def generate_time_series(n_samples=1000):
    """
    Generate synthetic time series with known causal structure:
    X(t) = 0.6*X(t-1) + noise
    Y(t) = 0.5*Y(t-1) + 0.4*X(t-2) + noise
    Z(t) = 0.3*Y(t-1) + noise
    
    Causal structure: X --2--> Y --1--> Z
    """
    data = np.zeros((n_samples, 3))
    
    for t in range(2, n_samples):
        # X: autoregressive process
        data[t, 0] = 0.6 * data[t-1, 0] + np.random.randn() * 0.5
        
        # Y: influenced by X at lag 2
        data[t, 1] = 0.5 * data[t-1, 1] + 0.4 * data[t-2, 0] + np.random.randn() * 0.5
        
        # Z: influenced by Y at lag 1
        data[t, 2] = 0.3 * data[t-1, 1] + np.random.randn() * 0.5
    
    return data


def visualize_time_series(data, var_names):
    """Visualize the generated time series"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    
    for i, (ax, name) in enumerate(zip(axes, var_names)):
        ax.plot(data[:200, i], linewidth=0.8)
        ax.set_ylabel(name, fontsize=12)
        ax.grid(alpha=0.3)
    
    axes[0].set_title('Synthetic Time Series (first 200 steps)', fontsize=14)
    axes[2].set_xlabel('Time', fontsize=12)
    plt.tight_layout()
    plt.show()


def run_pcmci(data, var_names, tau_max=5):
    """
    Run PCMCI algorithm to discover causal relationships
    
    Args:
        data: time series data (n_samples, n_vars)
        var_names: list of variable names
        tau_max: maximum time lag to consider
    """
    # Create dataframe object
    dataframe = pp.DataFrame(data, var_names=var_names)
    
    # Initialize PCMCI with ParCorr (partial correlation) test
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
    
    # Run PCMCI algorithm
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=0.05)
    
    return pcmci, results


def visualize_causal_graph(results, var_names):
    """Visualize the discovered causal graph"""
    tp.plot_graph(
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=var_names,
        link_colorbar_label='cross-correlation',
        node_colorbar_label='auto-correlation',
        figsize=(8, 6)
    )
    plt.show()


def print_results(results, var_names):
    """Print detected causal links"""
    print("DETECTED CAUSAL LINKS (p < 0.05)")
    
    graph = results['graph']
    p_matrix = results['p_matrix']
    
    # Print significant links
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            for tau in range(graph.shape[2]):
                if graph[i, j, tau] == '-->':
                    print(f"{var_names[i]} --{tau}--> {var_names[j]}"
                          f"  (p-value: {p_matrix[i, j, tau]:.4f})")
    

if __name__ == "__main__":
    # Configuration
    var_names = ['X', 'Y', 'Z']
    
    # Step 1: Generate data
    data = generate_time_series(n_samples=1000)
    visualize_time_series(data, var_names)
    
    # Step 2: Run PCMCI
    pcmci, results = run_pcmci(data, var_names, tau_max=5)
    visualize_causal_graph(results, var_names)
    
    # Step 3: Print results
    print_results(results, var_names)
