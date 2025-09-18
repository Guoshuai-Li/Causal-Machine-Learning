import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class FrontdoorCounterfactualPractice:
    def __init__(self):
        self.scaler = StandardScaler()
        self.dag = None
        self.data = None
        self.structural_equations = {}
        
    def generate_frontdoor_scenario(self, n=1000):
        """Generate front-door scenario: X → Z → Y with unmeasured U → X, U → Y"""
        # Generate unmeasured confounder U
        U = np.random.normal(0, 1, n)
        
        # Generate noise terms
        noise_x = np.random.normal(0, 0.5, n)
        noise_z = np.random.normal(0, 0.5, n)
        noise_y = np.random.normal(0, 0.5, n)
        
        # Structural equations with unmeasured confounding
        X = 0.8 * U + noise_x  # U → X (confounding)
        Z = 1.2 * X + noise_z  # X → Z (front-door pathway)
        Y = 0.7 * Z + 0.9 * U + noise_y  # Z → Y, U → Y (confounding)
        
        # Store data (U is unmeasured in practice)
        self.data = pd.DataFrame({
            'X': X, 'Z': Z, 'Y': Y, 'U': U  # U included for validation only
        })
        
        # Store structural equations for counterfactual generation
        self.structural_equations = {
            'noise_x': noise_x, 'noise_z': noise_z, 'noise_y': noise_y,
            'U': U  # Unmeasured but needed for counterfactuals
        }
        
        # Create DAG structure
        self.dag = nx.DiGraph()
        edges = [('U', 'X'), ('U', 'Y'), ('X', 'Z'), ('Z', 'Y')]
        self.dag.add_edges_from(edges)
        
        return self.data
    
    def visualize_frontdoor_dag(self):
        """Visualize front-door DAG with unmeasured confounder"""
        plt.figure(figsize=(10, 6))
        
        # Define positions
        pos = {
            'U': (1, 2), 'X': (0, 1), 'Z': (1, 1), 'Y': (2, 1)
        }
        
        # Draw measured variables
        measured_nodes = ['X', 'Z', 'Y']
        nx.draw_networkx_nodes(self.dag, pos, nodelist=measured_nodes, 
                              node_color='lightblue', node_size=1500, alpha=0.8, label='Measured Variables')
        
        # Draw unmeasured variable
        unmeasured_nodes = ['U']
        nx.draw_networkx_nodes(self.dag, pos, nodelist=unmeasured_nodes,
                              node_color='lightcoral', node_size=1500, alpha=0.8, label='Unmeasured Confounder')
        
        # Draw edges
        measured_edges = [('X', 'Z'), ('Z', 'Y')]
        unmeasured_edges = [('U', 'X'), ('U', 'Y')]
        
        nx.draw_networkx_edges(self.dag, pos, edgelist=measured_edges,
                              edge_color='blue', arrows=True, arrowsize=20)
        nx.draw_networkx_edges(self.dag, pos, edgelist=unmeasured_edges,
                              edge_color='red', style='dashed', arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(self.dag, pos, font_size=12, font_weight='bold')
        
        plt.title('Front-door Scenario: X → Z → Y with Unmeasured U → X, U → Y', 
                 fontsize=14, fontweight='bold')
        
        # Create custom legend with proper colors and line styles
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.8, label='Measured Variables'),
            Patch(facecolor='lightcoral', alpha=0.8, label='Unmeasured Confounder'),
            Line2D([0], [0], color='blue', lw=2, label='Observed Edges'),
            Line2D([0], [0], color='red', lw=2, linestyle='--', label='Hidden Confounding')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def frontdoor_identification(self):
        """Implement front-door identification: P(Y|do(X)) via P(Y|do(X)) = Σz P(Z=z|X) * P(Y|Z=z)"""
        results = {}
        
        # Direct estimation (biased due to unmeasured confounding)
        lr_direct = LinearRegression()
        lr_direct.fit(self.data[['X']], self.data['Y'])
        direct_effect = lr_direct.coef_[0]
        
        # Front-door formula implementation
        # Step 1: Estimate P(Z|X) - this is unbiased because no confounding on X→Z edge
        lr_xz = LinearRegression()
        lr_xz.fit(self.data[['X']], self.data['Z'])
        
        # Step 2: Estimate P(Y|Z,X) - we need to adjust for X to get P(Y|Z)
        lr_yzx = LinearRegression()
        lr_yzx.fit(self.data[['Z', 'X']], self.data['Y'])
        z_coef = lr_yzx.coef_[0]  # Effect of Z on Y
        
        # Step 3: Front-door calculation
        # For linear case: E[Y|do(X)] = effect_XZ * effect_ZY
        effect_xz = lr_xz.coef_[0]  # X → Z effect
        frontdoor_effect = effect_xz * z_coef  # Total indirect effect
        
        # Alternative: Numerical integration approach
        x_values = np.linspace(self.data['X'].min(), self.data['X'].max(), 100)
        frontdoor_effects = []
        
        for x_val in x_values:
            # Predict Z given X=x_val
            z_pred = lr_xz.predict([[x_val]])[0]
            
            # Predict Y given Z=z_pred, marginalizing over X
            # P(Y|Z=z) = Σx P(Y|Z=z,X=x) * P(X=x)
            y_effects = []
            for x_marg in x_values:
                y_pred = lr_yzx.predict([[z_pred, x_marg]])[0]
                y_effects.append(y_pred)
            
            frontdoor_effects.append(np.mean(y_effects))
        
        numerical_frontdoor = np.mean(np.diff(frontdoor_effects) / np.diff(x_values))
        
        # Ground truth (using unmeasured U for validation)
        # True effect: X→Z (1.2) * Z→Y (0.7) = 0.84
        ground_truth = 1.2 * 0.7
        
        results = {
            'direct_biased': direct_effect,
            'frontdoor_analytical': frontdoor_effect,
            'frontdoor_numerical': numerical_frontdoor,
            'ground_truth': ground_truth,
            'x_to_z_effect': effect_xz,
            'z_to_y_effect': z_coef
        }
        
        return results
    
    def counterfactual_reasoning(self):
        """Implement counterfactual reasoning using Abduction-Action-Prediction"""
        results = {}
        
        # Select a specific individual for counterfactual analysis
        individual_idx = 100
        observed_x = self.data.iloc[individual_idx]['X']
        observed_z = self.data.iloc[individual_idx]['Z']  
        observed_y = self.data.iloc[individual_idx]['Y']
        observed_u = self.data.iloc[individual_idx]['U']  # Normally unmeasured
        
        # Step 1: Abduction - infer noise terms from observed data
        # From structural equations:
        # X = 0.8*U + εx → εx = X - 0.8*U
        # Z = 1.2*X + εz → εz = Z - 1.2*X  
        # Y = 0.7*Z + 0.9*U + εy → εy = Y - 0.7*Z - 0.9*U
        
        inferred_noise_x = observed_x - 0.8 * observed_u
        inferred_noise_z = observed_z - 1.2 * observed_x
        inferred_noise_y = observed_y - 0.7 * observed_z - 0.9 * observed_u
        
        # Step 2: Action - intervene on X (set X to counterfactual value)
        counterfactual_x_values = [observed_x - 1, observed_x + 1]
        counterfactual_results = []
        
        for cf_x in counterfactual_x_values:
            # Step 3: Prediction - compute counterfactual outcomes
            # X is set by intervention
            cf_x_val = cf_x
            
            # Z follows structural equation with intervened X
            cf_z = 1.2 * cf_x_val + inferred_noise_z
            
            # Y follows structural equation with counterfactual Z  
            cf_y = 0.7 * cf_z + 0.9 * observed_u + inferred_noise_y
            
            counterfactual_results.append({
                'cf_x': cf_x_val,
                'cf_z': cf_z, 
                'cf_y': cf_y
            })
        
        # Individual treatment effect
        ite = counterfactual_results[1]['cf_y'] - counterfactual_results[0]['cf_y']
        
        results = {
            'individual_idx': individual_idx,
            'observed': {'x': observed_x, 'z': observed_z, 'y': observed_y},
            'counterfactuals': counterfactual_results,
            'individual_treatment_effect': ite,
            'inferred_noises': {
                'noise_x': inferred_noise_x,
                'noise_z': inferred_noise_z, 
                'noise_y': inferred_noise_y
            }
        }
        
        return results
    
    def potential_outcomes_framework(self):
        """Demonstrate SCM-Potential Outcomes bridge using do-notation"""
        results = {}
        
        # Generate potential outcomes for different treatment levels
        treatment_levels = [0, 1, 2]
        potential_outcomes = {}
        
        n = len(self.data)
        U = self.structural_equations['U']
        noise_z = self.structural_equations['noise_z']
        noise_y = self.structural_equations['noise_y']
        
        for t in treatment_levels:
            # Under intervention do(X=t)
            X_t = np.full(n, t)  # X set to treatment level
            Z_t = 1.2 * X_t + noise_z  # Z follows structural equation
            Y_t = 0.7 * Z_t + 0.9 * U + noise_y  # Y follows structural equation
            
            potential_outcomes[f'Y_{t}'] = Y_t
        
        # Calculate Average Treatment Effects (ATE)
        ate_1_0 = potential_outcomes['Y_1'].mean() - potential_outcomes['Y_0'].mean()
        ate_2_0 = potential_outcomes['Y_2'].mean() - potential_outcomes['Y_0'].mean()
        ate_2_1 = potential_outcomes['Y_2'].mean() - potential_outcomes['Y_1'].mean()
        
        # Verify linearity assumption
        expected_ate_2_0 = 2 * ate_1_0  # Should equal ate_2_0 if linear
        
        results = {
            'potential_outcomes_means': {
                'E[Y_0]': potential_outcomes['Y_0'].mean(),
                'E[Y_1]': potential_outcomes['Y_1'].mean(), 
                'E[Y_2]': potential_outcomes['Y_2'].mean()
            },
            'treatment_effects': {
                'ATE(1,0)': ate_1_0,
                'ATE(2,0)': ate_2_0,
                'ATE(2,1)': ate_2_1
            },
            'linearity_check': {
                'expected_2*ATE(1,0)': expected_ate_2_0,
                'actual_ATE(2,0)': ate_2_0,
                'difference': abs(expected_ate_2_0 - ate_2_0)
            },
            'ground_truth_ate': 1.2 * 0.7  # True causal effect X→Z→Y
        }
        
        return results
    
    def mechanism_invariance_test(self):
        """Test mechanism invariance: P(Y|X) vs P(Y|do(X)) and distribution changes"""
        results = {}
        
        # Original distribution
        original_effect = LinearRegression()
        original_effect.fit(self.data[['X']], self.data['Y'])
        observational_effect = original_effect.coef_[0]
        
        # Simulate intervention do(X) by changing P(X) distribution
        n = len(self.data)
        U = self.structural_equations['U']
        noise_z = self.structural_equations['noise_z']  
        noise_y = self.structural_equations['noise_y']
        
        # Change X distribution (but keep structural equations)
        X_new_dist = np.random.exponential(1, n) - 1  # Different distribution
        Z_new = 1.2 * X_new_dist + noise_z  # Same structural equation
        Y_new = 0.7 * Z_new + 0.9 * U + noise_y  # Same structural equation
        
        # Estimate effect under new distribution
        new_dist_effect = LinearRegression()
        new_dist_effect.fit(X_new_dist.reshape(-1, 1), Y_new)
        new_dist_observational = new_dist_effect.coef_[0]
        
        # True interventional effect (should be invariant)
        interventional_effect = 1.2 * 0.7  # X→Z→Y pathway
        
        results = {
            'original_observational': observational_effect,
            'new_distribution_observational': new_dist_observational,
            'true_interventional': interventional_effect,
            'mechanism_invariance_test': {
                'original_bias': abs(observational_effect - interventional_effect),
                'new_dist_bias': abs(new_dist_observational - interventional_effect),
                'bias_stability': abs(observational_effect - new_dist_observational)
            }
        }
        
        return results
    
    def demonstrate_confounding_failure(self):
        """Show why backdoor adjustment fails but front-door works"""
        results = {}
        
        # Attempt backdoor adjustment (should fail due to unmeasured U)
        # Only observed variables available: X, Z, Y
        
        # Method 1: Naive regression Y ~ X (biased)
        naive = LinearRegression()
        naive.fit(self.data[['X']], self.data['Y'])
        naive_effect = naive.coef_[0]
        
        # Method 2: "Adjustment" for Z (wrong - Z is mediator, not confounder)
        wrong_adj = LinearRegression() 
        wrong_adj.fit(self.data[['X', 'Z']], self.data['Y'])
        wrong_effect = wrong_adj.coef_[0]  # This blocks the causal pathway!
        
        # Method 3: Front-door identification (correct)
        frontdoor_results = self.frontdoor_identification()
        frontdoor_effect = frontdoor_results['frontdoor_analytical']
        
        # Method 4: Oracle adjustment with unmeasured U (theoretical best)
        oracle = LinearRegression()
        oracle.fit(self.data[['X', 'U']], self.data['Y'])
        oracle_effect = oracle.coef_[0]  # Still biased due to X-Z-Y pathway
        
        # True total effect
        true_effect = 1.2 * 0.7  # X→Z→Y
        
        results = {
            'naive_regression': naive_effect,
            'wrong_adjustment_for_Z': wrong_effect,
            'frontdoor_identification': frontdoor_effect,
            'oracle_with_U': oracle_effect,
            'ground_truth': true_effect,
            'biases': {
                'naive_bias': abs(naive_effect - true_effect),
                'wrong_adj_bias': abs(wrong_effect - true_effect),
                'frontdoor_bias': abs(frontdoor_effect - true_effect),
                'oracle_bias': abs(oracle_effect - true_effect)
            }
        }
        
        return results
    
    def run_complete_experiment(self):
        """Run complete experimental pipeline"""
        
        # 1. Generate front-door scenario
        print("1. Generating front-door scenario with unmeasured confounding")
        data = self.generate_frontdoor_scenario()
        print(f"   Generated data with X→Z→Y pathway and unmeasured U→X,Y confounding")
        print(f"   Sample size: {len(data)} observations")
        
        # 2. Visualize DAG
        print("\n2. Visualizing front-door DAG structure")
        self.visualize_frontdoor_dag()
        
        # 3. Front-door identification
        print("\n3. Front-door identification analysis:")
        frontdoor_results = self.frontdoor_identification()
        print(f"   Direct biased estimate: {frontdoor_results['direct_biased']:.3f}")
        print(f"   Front-door analytical: {frontdoor_results['frontdoor_analytical']:.3f}")
        print(f"   Front-door numerical: {frontdoor_results['frontdoor_numerical']:.3f}")
        print(f"   Ground truth effect: {frontdoor_results['ground_truth']:.3f}")
        print(f"   X→Z effect: {frontdoor_results['x_to_z_effect']:.3f}")
        print(f"   Z→Y effect: {frontdoor_results['z_to_y_effect']:.3f}")
        
        # 4. Counterfactual reasoning
        print("\n4. Counterfactual reasoning (Abduction-Action-Prediction):")
        cf_results = self.counterfactual_reasoning()
        print(f"   Individual {cf_results['individual_idx']} observed: X={cf_results['observed']['x']:.3f}, Y={cf_results['observed']['y']:.3f}")
        for i, cf in enumerate(cf_results['counterfactuals']):
            print(f"   Counterfactual {i+1}: X={cf['cf_x']:.3f} → Z={cf['cf_z']:.3f} → Y={cf['cf_y']:.3f}")
        print(f"   Individual treatment effect: {cf_results['individual_treatment_effect']:.3f}")
        
        # 5. Potential outcomes framework
        print("\n5. SCM-Potential Outcomes (PO) framework bridge:")
        po_results = self.potential_outcomes_framework()
        print("   Potential outcome means:")
        for po, mean_val in po_results['potential_outcomes_means'].items():
            print(f"   {po}: {mean_val:.3f}")
        print("   Average treatment effects:")
        for ate, effect in po_results['treatment_effects'].items():
            print(f"   {ate}: {effect:.3f}")
        print("   Linearity check:")
        print(f"   Expected 2*ATE(1,0): {po_results['linearity_check']['expected_2*ATE(1,0)']:.3f}")
        print(f"   Actual ATE(2,0): {po_results['linearity_check']['actual_ATE(2,0)']:.3f}")
        print(f"   Difference: {po_results['linearity_check']['difference']:.3f}")
        
        # 6. Mechanism invariance test
        print("\n6. Mechanism invariance under distribution changes:")
        invariance_results = self.mechanism_invariance_test()
        print(f"   Original observational effect: {invariance_results['original_observational']:.3f}")
        print(f"   New distribution observational: {invariance_results['new_distribution_observational']:.3f}")
        print(f"   True interventional effect: {invariance_results['true_interventional']:.3f}")
        print(f"   Original bias: {invariance_results['mechanism_invariance_test']['original_bias']:.3f}")
        print(f"   New distribution bias: {invariance_results['mechanism_invariance_test']['new_dist_bias']:.3f}")
        print(f"   Bias stability: {invariance_results['mechanism_invariance_test']['bias_stability']:.3f}")
        
        # 7. Confounding failure demonstration  
        print("\n7. Why backdoor fails but front-door works:")
        failure_results = self.demonstrate_confounding_failure()
        print(f"   Naive regression Y~X: {failure_results['naive_regression']:.3f}")
        print(f"   Wrong adjustment for Z: {failure_results['wrong_adjustment_for_Z']:.3f}")
        print(f"   Front-door identification: {failure_results['frontdoor_identification']:.3f}")
        print(f"   Oracle with unmeasured U: {failure_results['oracle_with_U']:.3f}")
        print(f"   Ground truth: {failure_results['ground_truth']:.3f}")
        print("   Bias magnitudes:")
        for method, bias in failure_results['biases'].items():
            print(f"   {method}: {bias:.3f}")
        
        # Store results for further analysis
        self.experiment_results = {
            'frontdoor': frontdoor_results,
            'counterfactual': cf_results,
            'potential_outcomes': po_results,
            'mechanism_invariance': invariance_results,
            'method_comparison': failure_results
        }

# Run experiment
if __name__ == "__main__":
    experiment = FrontdoorCounterfactualPractice()
    experiment.run_complete_experiment()
