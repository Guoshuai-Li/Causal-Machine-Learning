import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class TimeSeriesCausalInference:
    def __init__(self):
        self.data = None
        self.results = {}
        
    def generate_time_series_dag(self, n_timesteps=500):
        """Generate time series data following a temporal causal DAG"""
        # Initialize time series
        X = np.zeros(n_timesteps)
        Y = np.zeros(n_timesteps)
        Z = np.zeros(n_timesteps)
        
        # Set initial values
        X[0] = np.random.normal(0, 1)
        Y[0] = np.random.normal(0, 1)
        Z[0] = np.random.normal(0, 1)
        
        # Generate time series with temporal causal relationships
        for t in range(1, n_timesteps):
            # X_t depends on its own history
            X[t] = 0.6 * X[t-1] + np.random.normal(0, 0.5)
            
            # Y_t depends on X_{t-1} and its own history (X causes Y with lag 1)
            Y[t] = 0.4 * Y[t-1] + 0.8 * X[t-1] + np.random.normal(0, 0.3)
            
            # Z_t depends on Y_{t-2} and its own history (Y causes Z with lag 2)
            if t >= 2:
                Z[t] = 0.5 * Z[t-1] + 0.6 * Y[t-2] + np.random.normal(0, 0.4)
            else:
                Z[t] = 0.5 * Z[t-1] + np.random.normal(0, 0.4)
        
        # Store data
        self.data = pd.DataFrame({
            'X': X,
            'Y': Y, 
            'Z': Z,
            'time': range(n_timesteps)
        })
        
        # True causal relationships for validation
        self.true_relationships = {
            'X_causes_Y_lag1': True,
            'Y_causes_Z_lag2': True,
            'X_causes_Z_direct': False,  # Only indirect through Y
            'instantaneous_effects': False
        }
        
        return self.data
    
    def granger_causality_analysis(self, max_lag=5):
        """Perform Granger causality tests between all variable pairs"""
        variables = ['X', 'Y', 'Z']
        results = {}
        
        # Test all pairs of variables
        for cause_var in variables:
            for effect_var in variables:
                if cause_var != effect_var:
                    # Prepare data for Granger test
                    test_data = self.data[[cause_var, effect_var]].dropna()
                    
                    try:
                        # Perform Granger causality test
                        gc_result = grangercausalitytests(
                            test_data[[effect_var, cause_var]], 
                            maxlag=max_lag, 
                            verbose=False
                        )
                        
                        # Extract p-values for different lags
                        p_values = []
                        for lag in range(1, max_lag + 1):
                            if lag in gc_result:
                                p_val = gc_result[lag][0]['ssr_ftest'][1]
                                p_values.append(p_val)
                        
                        # Find best lag (minimum p-value)
                        if p_values:
                            min_p_idx = np.argmin(p_values)
                            best_lag = min_p_idx + 1
                            min_p_value = p_values[min_p_idx]
                            
                            results[f'{cause_var}_causes_{effect_var}'] = {
                                'best_lag': best_lag,
                                'p_value': min_p_value,
                                'significant': min_p_value < 0.05,
                                'all_p_values': p_values
                            }
                    except:
                        results[f'{cause_var}_causes_{effect_var}'] = {
                            'best_lag': None,
                            'p_value': 1.0,
                            'significant': False,
                            'all_p_values': []
                        }
        
        return results
    
    def var_model_analysis(self):
        """Analyze time series using Vector Autoregression (VAR) model"""
        # Prepare data for VAR
        var_data = self.data[['X', 'Y', 'Z']].dropna()
        
        # Fit VAR model
        try:
            model = VAR(var_data)
            # Select optimal lag using AIC
            lag_order = model.select_order(maxlags=10)
            optimal_lag = lag_order.aic
            
            # Fit VAR with optimal lag
            fitted_model = model.fit(optimal_lag)
            
            # Extract coefficients
            coefficients = fitted_model.coefs
            variable_names = ['X', 'Y', 'Z']
            
            # Organize results
            var_results = {
                'optimal_lag': optimal_lag,
                'coefficients': {},
                'significant_relationships': []
            }
            
            # Extract significant relationships
            for lag in range(optimal_lag):
                for i, target_var in enumerate(variable_names):
                    for j, source_var in enumerate(variable_names):
                        coef = coefficients[lag][i, j]
                        # Use fitted model's standard errors if available
                        try:
                            stderr = fitted_model.stderr[lag][i, j]
                            t_stat = coef / stderr if stderr > 0 else 0
                            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
                            
                            relationship = {
                                'source': source_var,
                                'target': target_var,
                                'lag': lag + 1,
                                'coefficient': coef,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                            
                            var_results['coefficients'][f'{source_var}_to_{target_var}_lag{lag+1}'] = relationship
                            
                            if relationship['significant'] and abs(coef) > 0.1:
                                var_results['significant_relationships'].append(relationship)
                        except:
                            # Fallback if standard errors not available
                            if abs(coef) > 0.1:
                                relationship = {
                                    'source': source_var,
                                    'target': target_var,
                                    'lag': lag + 1,
                                    'coefficient': coef,
                                    'p_value': None,
                                    'significant': True
                                }
                                var_results['significant_relationships'].append(relationship)
            
            return var_results
            
        except Exception as e:
            return {
                'optimal_lag': 1,
                'coefficients': {},
                'significant_relationships': [],
                'error': str(e)
            }
    
    def graph_based_methods(self):
        """Apply graph-based causal discovery to time series data"""
        # Convert time series to lagged feature matrix
        def create_lagged_features(data, max_lag=3):
            """Create lagged features for causal discovery"""
            variables = ['X', 'Y', 'Z']
            lagged_data = {}
            
            for var in variables:
                for lag in range(max_lag + 1):
                    if lag == 0:
                        lagged_data[f'{var}_t'] = data[var].values[max_lag:]
                    else:
                        lagged_data[f'{var}_t-{lag}'] = data[var].values[max_lag-lag:-lag]
            
            return pd.DataFrame(lagged_data)
        
        lagged_df = create_lagged_features(self.data, max_lag=3)
        
        # Simplified PC algorithm for time series
        def simplified_pc_timeseries(data, alpha=0.05):
            """Simplified PC algorithm adapted for time series"""
            variables = list(data.columns)
            current_vars = [var for var in variables if '_t' in var and 't-' not in var]  # Current time variables
            past_vars = [var for var in variables if 't-' in var]  # Lagged variables
            
            significant_edges = []
            
            # Test relationships from past to current
            for past_var in past_vars:
                for current_var in current_vars:
                    # Simple correlation test
                    corr, p_value = stats.pearsonr(data[past_var], data[current_var])
                    
                    if p_value < alpha and abs(corr) > 0.1:
                        # Extract lag from variable name
                        if 't-' in past_var:
                            lag = int(past_var.split('t-')[1])
                        else:
                            lag = 0
                            
                        base_past_var = past_var.split('_t')[0]
                        base_current_var = current_var.split('_t')[0]
                        
                        significant_edges.append({
                            'source': base_past_var,
                            'target': base_current_var,
                            'lag': lag,
                            'correlation': corr,
                            'p_value': p_value
                        })
            
            return significant_edges
        
        # Apply simplified PC algorithm
        graph_edges = simplified_pc_timeseries(lagged_df)
        
        # Compare with Granger causality results
        graph_results = {
            'discovered_edges': graph_edges,
            'num_edges': len(graph_edges)
        }
        
        return graph_results
    
    def dynamic_causal_experiment(self, intervention_start=250, intervention_end=350):
        """Simulate dynamic causal intervention experiment"""
        # Generate baseline time series
        baseline_data = self.generate_time_series_dag(n_timesteps=500)
        
        # Generate intervened time series
        n_timesteps = 500
        X_int = np.zeros(n_timesteps)
        Y_int = np.zeros(n_timesteps)
        Z_int = np.zeros(n_timesteps)
        
        # Set initial values
        X_int[0] = np.random.normal(0, 1)
        Y_int[0] = np.random.normal(0, 1)
        Z_int[0] = np.random.normal(0, 1)
        
        # Generate with intervention on X during specified period
        for t in range(1, n_timesteps):
            # Intervention: set X to higher value during intervention period
            if intervention_start <= t <= intervention_end:
                X_int[t] = 2.0  # Fixed high value (intervention)
            else:
                X_int[t] = 0.6 * X_int[t-1] + np.random.normal(0, 0.5)
            
            # Y and Z follow same structural equations
            Y_int[t] = 0.4 * Y_int[t-1] + 0.8 * X_int[t-1] + np.random.normal(0, 0.3)
            
            if t >= 2:
                Z_int[t] = 0.5 * Z_int[t-1] + 0.6 * Y_int[t-2] + np.random.normal(0, 0.4)
            else:
                Z_int[t] = 0.5 * Z_int[t-1] + np.random.normal(0, 0.4)
        
        intervened_data = pd.DataFrame({
            'X': X_int,
            'Y': Y_int,
            'Z': Z_int,
            'time': range(n_timesteps)
        })
        
        # Analysis of intervention effects
        intervention_results = {
            'intervention_period': (intervention_start, intervention_end),
            'baseline_means': {
                'X': baseline_data['X'].mean(),
                'Y': baseline_data['Y'].mean(),
                'Z': baseline_data['Z'].mean()
            },
            'intervention_means': {
                'X': intervened_data['X'][intervention_start:intervention_end].mean(),
                'Y': intervened_data['Y'][intervention_start:intervention_end].mean(),
                'Z': intervened_data['Z'][intervention_start:intervention_end].mean()
            }
        }
        
        # Calculate effect sizes
        for var in ['Y', 'Z']:
            baseline_mean = baseline_data[var].mean()
            intervention_mean = intervened_data[var][intervention_start:intervention_end].mean()
            effect_size = intervention_mean - baseline_mean
            intervention_results[f'{var}_effect_size'] = effect_size
        
        return baseline_data, intervened_data, intervention_results
    
    def visualize_results(self):
        """Visualize time series and causal analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Original time series
        time = self.data['time']
        axes[0,0].plot(time, self.data['X'], label='X', alpha=0.7)
        axes[0,0].plot(time, self.data['Y'], label='Y', alpha=0.7)
        axes[0,0].plot(time, self.data['Z'], label='Z', alpha=0.7)
        axes[0,0].set_title('Generated Time Series')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Value')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Granger causality results
        if 'granger_results' in self.results:
            gc_results = self.results['granger_results']
            relationships = []
            p_values = []
            colors = []
            
            for rel_name, result in gc_results.items():
                relationships.append(rel_name.replace('_causes_', ' → '))
                p_values.append(-np.log10(result['p_value'] + 1e-10))  # -log10 p-value
                colors.append('green' if result['significant'] else 'red')
            
            y_pos = np.arange(len(relationships))
            axes[0,1].barh(y_pos, p_values, color=colors, alpha=0.7)
            axes[0,1].set_yticks(y_pos)
            axes[0,1].set_yticklabels(relationships)
            axes[0,1].set_xlabel('-log10(p-value)')
            axes[0,1].set_title('Granger Causality Test Results')
            axes[0,1].axvline(x=-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
        
        # Plot 3: Dynamic intervention effects
        if 'intervention_results' in self.results:
            baseline_data, intervened_data, int_results = self.results['intervention_results']
            
            # Plot Y variable (should show immediate response to X intervention)
            axes[1,0].plot(baseline_data['time'], baseline_data['Y'], 
                          label='Baseline Y', alpha=0.7, color='blue')
            axes[1,0].plot(intervened_data['time'], intervened_data['Y'], 
                          label='Intervened Y', alpha=0.7, color='red')
            
            # Mark intervention period
            int_start, int_end = int_results['intervention_period']
            axes[1,0].axvspan(int_start, int_end, alpha=0.2, color='gray', 
                             label='Intervention Period')
            
            axes[1,0].set_title('Effect of X Intervention on Y')
            axes[1,0].set_xlabel('Time')
            axes[1,0].set_ylabel('Y Value')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: VAR model coefficients
        if 'var_results' in self.results:
            var_results = self.results['var_results']
            if 'significant_relationships' in var_results:
                sig_rels = var_results['significant_relationships']
                
                if sig_rels:
                    sources = [rel['source'] for rel in sig_rels]
                    targets = [rel['target'] for rel in sig_rels]
                    coeffs = [abs(rel['coefficient']) for rel in sig_rels]
                    lags = [rel['lag'] for rel in sig_rels]
                    
                    # Create network-style visualization
                    G = nx.DiGraph()
                    for i, (s, t, c, l) in enumerate(zip(sources, targets, coeffs, lags)):
                        G.add_edge(f'{s}', f'{t}', weight=c, lag=l)
                    
                    if len(G.nodes()) > 0:
                        pos = nx.spring_layout(G)
                        nx.draw_networkx_nodes(G, pos, ax=axes[1,1], 
                                             node_color='lightblue', node_size=1000)
                        nx.draw_networkx_labels(G, pos, ax=axes[1,1])
                        
                        # Draw edges with thickness proportional to coefficient
                        for (s, t, d) in G.edges(data=True):
                            axes[1,1].annotate('', xy=pos[t], xytext=pos[s],
                                             arrowprops=dict(arrowstyle='->', 
                                                           lw=d['weight']*5,
                                                           color='blue'))
                        
                        axes[1,1].set_title('VAR Model Significant Relationships')
                        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_experiment(self):
        """Run complete time series causal inference experimental pipeline"""
        print("=== Week 8: Time Series Causal Inference ===\n")
        
        # 1. Generate temporal causal DAG
        print("1. Generating time series data with temporal causal structure...")
        data = self.generate_time_series_dag(n_timesteps=500)
        print(f"   Generated time series with {len(data)} time points")
        print("   True causal relationships:")
        for rel, val in self.true_relationships.items():
            print(f"   {rel}: {val}")
        
        # 2. Granger causality analysis
        print("\n2. Granger causality analysis:")
        granger_results = self.granger_causality_analysis(max_lag=5)
        self.results['granger_results'] = granger_results
        
        for relationship, result in granger_results.items():
            print(f"   {relationship}:")
            print(f"   Best lag: {result['best_lag']}, p-value: {result['p_value']:.4f}")
            print(f"   Significant: {result['significant']}")
        
        # 3. VAR model analysis
        print("\n3. Vector Autoregression (VAR) model analysis:")
        var_results = self.var_model_analysis()
        self.results['var_results'] = var_results
        
        print(f"   Optimal lag order: {var_results['optimal_lag']}")
        print(f"   Number of significant relationships: {len(var_results['significant_relationships'])}")
        
        if var_results['significant_relationships']:
            print("   Significant relationships found:")
            for rel in var_results['significant_relationships']:
                print(f"   {rel['source']} → {rel['target']} (lag {rel['lag']}): coef={rel['coefficient']:.3f}")
        
        # 4. Graph-based causal discovery
        print("\n4. Graph-based methods for time series:")
        graph_results = self.graph_based_methods()
        self.results['graph_results'] = graph_results
        
        print(f"   Discovered {graph_results['num_edges']} potential causal edges:")
        for edge in graph_results['discovered_edges']:
            print(f"   {edge['source']} → {edge['target']} (lag {edge['lag']}): r={edge['correlation']:.3f}")
        
        # 5. Dynamic causal experiment
        print("\n5. Dynamic causal intervention experiment:")
        baseline_data, intervened_data, intervention_results = self.dynamic_causal_experiment()
        self.results['intervention_results'] = (baseline_data, intervened_data, intervention_results)
        
        int_start, int_end = intervention_results['intervention_period']
        print(f"   Intervention period: t={int_start} to t={int_end}")
        print("   Baseline means:", {k: f"{v:.3f}" for k, v in intervention_results['baseline_means'].items()})
        print("   Intervention means:", {k: f"{v:.3f}" for k, v in intervention_results['intervention_means'].items()})
        
        for var in ['Y', 'Z']:
            effect = intervention_results[f'{var}_effect_size']
            print(f"   Effect on {var}: {effect:.3f}")
        
        # 6. Visualize results
        print("\n6. Visualizing time series causal analysis results...")
        self.visualize_results()
        
        # 7. Summary and validation
        print("\n7. Method comparison and validation:")
        
        # Compare Granger vs Graph methods
        granger_xy = granger_results.get('X_causes_Y', {}).get('significant', False)
        granger_yz = granger_results.get('Y_causes_Z', {}).get('significant', False)
        
        graph_xy = any(edge['source'] == 'X' and edge['target'] == 'Y' 
                      for edge in graph_results['discovered_edges'])
        graph_yz = any(edge['source'] == 'Y' and edge['target'] == 'Z'
                      for edge in graph_results['discovered_edges'])
        
        print(f"   X causes Y - Granger: {granger_xy}, Graph: {graph_xy}, True: {self.true_relationships['X_causes_Y_lag1']}")
        print(f"   Y causes Z - Granger: {granger_yz}, Graph: {graph_yz}, True: {self.true_relationships['Y_causes_Z_lag2']}")
        
        # Validate intervention effects
        y_effect = intervention_results['Y_effect_size']
        z_effect = intervention_results['Z_effect_size'] 
        print(f"   Intervention validation - Y effect: {y_effect:.3f} (expected > 0)")
        print(f"   Intervention validation - Z effect: {z_effect:.3f} (expected > 0 with delay)")

# Run experiment
if __name__ == "__main__":
    experiment = TimeSeriesCausalInference()
    experiment.run_complete_experiment()
