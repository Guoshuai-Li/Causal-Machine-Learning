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
from itertools import combinations

np.random.seed(42)

class MultivariateInterventionPractice:
    def __init__(self):
        self.scaler = StandardScaler()
        self.dag = None
        self.data = None
        self.adjustment_sets = {}
        
    def generate_multivariate_dag(self, n=1000):
        """Generate multivariate DAG with 6-8 nodes following SCM framework"""
        # Create DAG structure: Z1 -> X -> Y, Z2 -> X, Z2 -> Y, Z3 -> Y
        # Additional relationships: Z1 -> Z2, creating more complex confounding
        
        # Generate exogenous noise terms
        noise_z1 = np.random.normal(0, 0.5, n)
        noise_z2 = np.random.normal(0, 0.5, n)  
        noise_z3 = np.random.normal(0, 0.5, n)
        noise_x = np.random.normal(0, 0.5, n)
        noise_y = np.random.normal(0, 0.5, n)
        noise_m = np.random.normal(0, 0.5, n)
        
        # Generate variables following causal structure
        Z1 = noise_z1
        Z2 = 0.6 * Z1 + noise_z2  # Z1 -> Z2
        Z3 = noise_z3
        X = 0.8 * Z1 + 0.7 * Z2 + noise_x  # Z1, Z2 -> X
        M = 0.5 * X + noise_m  # X -> M (mediator)
        Y = 1.0 * X + 0.6 * Z2 + 0.4 * Z3 + 0.3 * M + noise_y  # X, Z2, Z3, M -> Y
        
        # Store data
        self.data = pd.DataFrame({
            'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 
            'X': X, 'M': M, 'Y': Y
        })
        
        # Create networkx DAG for visualization and analysis
        self.dag = nx.DiGraph()
        edges = [('Z1', 'Z2'), ('Z1', 'X'), ('Z2', 'X'), ('Z2', 'Y'), 
                ('Z3', 'Y'), ('X', 'M'), ('X', 'Y'), ('M', 'Y')]
        self.dag.add_edges_from(edges)
        
        return self.data, self.dag
    
    def visualize_dag(self):
        """Visualize DAG structure with causal relationships"""
        plt.figure(figsize=(10, 8))
        
        # Create layout
        pos = {
            'Z1': (0, 2), 'Z2': (1, 2), 'Z3': (2, 2),
            'X': (0.5, 1), 'M': (1.5, 1), 'Y': (1, 0)
        }
        
        # Draw nodes
        nx.draw_networkx_nodes(self.dag, pos, node_color='lightblue', 
                              node_size=1500, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.dag, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, arrowstyle='->')
        
        # Draw labels
        nx.draw_networkx_labels(self.dag, pos, font_size=12, font_weight='bold')
        
        plt.title('Multivariate Causal DAG Structure', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return pos
    
    def intervention_vs_conditioning(self):
        """Compare P(Y|X) vs P(Y|do(X)) at non-source nodes"""
        X_values = [-1, 0, 1]  # Different intervention levels
        results = {}
        
        # Observational: P(Y|X)
        observational_means = []
        for x_val in X_values:
            mask = np.abs(self.data['X'] - x_val) < 0.2  # approximate conditioning
            if mask.sum() > 10:  # ensure sufficient data
                obs_mean = self.data.loc[mask, 'Y'].mean()
            else:
                obs_mean = np.nan
            observational_means.append(obs_mean)
        
        # Interventional: P(Y|do(X)) - simulate intervention
        interventional_means = []
        for x_val in X_values:
            # Under intervention do(X=x), we fix X and sample other variables naturally
            n_sim = 1000
            # Z1, Z2, Z3 remain unchanged (exogenous)
            Z1_int = np.random.normal(0, 0.5, n_sim)
            Z2_int = 0.6 * Z1_int + np.random.normal(0, 0.5, n_sim)
            Z3_int = np.random.normal(0, 0.5, n_sim)
            
            # X is fixed by intervention
            X_int = np.full(n_sim, x_val)
            
            # M depends on X
            M_int = 0.5 * X_int + np.random.normal(0, 0.5, n_sim)
            
            # Y depends on X, Z2, Z3, M
            Y_int = 1.0 * X_int + 0.6 * Z2_int + 0.4 * Z3_int + 0.3 * M_int + np.random.normal(0, 0.5, n_sim)
            
            interventional_means.append(Y_int.mean())
        
        results['observational'] = observational_means
        results['interventional'] = interventional_means
        results['x_values'] = X_values
        
        return results
    
    def demonstrate_source_node_equivalence(self):
        """Show P(Y|X) = P(Y|do(X)) at source nodes"""
        # Z1 is a source node (no parents)
        Z1_values = [-1, 0, 1]
        results = {}
        
        # For source nodes, conditioning and intervention should be equivalent
        observational_means = []
        for z_val in Z1_values:
            mask = np.abs(self.data['Z1'] - z_val) < 0.2
            if mask.sum() > 10:
                obs_mean = self.data.loc[mask, 'Y'].mean()
            else:
                obs_mean = np.nan
            observational_means.append(obs_mean)
        
        # Simulate intervention on source node
        interventional_means = []
        for z_val in Z1_values:
            n_sim = 1000
            # Z1 is fixed by intervention
            Z1_int = np.full(n_sim, z_val)
            
            # All other variables follow their structural equations
            Z2_int = 0.6 * Z1_int + np.random.normal(0, 0.5, n_sim)
            Z3_int = np.random.normal(0, 0.5, n_sim)
            X_int = 0.8 * Z1_int + 0.7 * Z2_int + np.random.normal(0, 0.5, n_sim)
            M_int = 0.5 * X_int + np.random.normal(0, 0.5, n_sim)
            Y_int = 1.0 * X_int + 0.6 * Z2_int + 0.4 * Z3_int + 0.3 * M_int + np.random.normal(0, 0.5, n_sim)
            
            interventional_means.append(Y_int.mean())
        
        results['observational'] = observational_means
        results['interventional'] = interventional_means
        results['z_values'] = Z1_values
        
        return results
    
    def multipoint_intervention_analysis(self):
        """Extended multipoint intervention P(Y|do(X,Z)) vs single point effects"""
        results = {}
        
        # Single interventions
        single_x_effect = []
        single_z2_effect = []
        
        # Baseline (no intervention)
        baseline_y = self.data['Y'].mean()
        
        # Single intervention do(X=1)
        n_sim = 1000
        Z1_sim = np.random.normal(0, 0.5, n_sim)
        Z2_sim = 0.6 * Z1_sim + np.random.normal(0, 0.5, n_sim)
        Z3_sim = np.random.normal(0, 0.5, n_sim)
        X_sim = np.full(n_sim, 1.0)  # do(X=1)
        M_sim = 0.5 * X_sim + np.random.normal(0, 0.5, n_sim)
        Y_sim_x = 1.0 * X_sim + 0.6 * Z2_sim + 0.4 * Z3_sim + 0.3 * M_sim + np.random.normal(0, 0.5, n_sim)
        single_x_mean = Y_sim_x.mean()
        
        # Single intervention do(Z2=1)
        Z1_sim = np.random.normal(0, 0.5, n_sim)
        Z2_sim = np.full(n_sim, 1.0)  # do(Z2=1)
        Z3_sim = np.random.normal(0, 0.5, n_sim)
        X_sim = 0.8 * Z1_sim + 0.7 * Z2_sim + np.random.normal(0, 0.5, n_sim)
        M_sim = 0.5 * X_sim + np.random.normal(0, 0.5, n_sim)
        Y_sim_z2 = 1.0 * X_sim + 0.6 * Z2_sim + 0.4 * Z3_sim + 0.3 * M_sim + np.random.normal(0, 0.5, n_sim)
        single_z2_mean = Y_sim_z2.mean()
        
        # Joint intervention do(X=1, Z2=1)
        Z1_sim = np.random.normal(0, 0.5, n_sim)
        Z2_sim = np.full(n_sim, 1.0)  # do(Z2=1)
        Z3_sim = np.random.normal(0, 0.5, n_sim)
        X_sim = np.full(n_sim, 1.0)  # do(X=1)
        M_sim = 0.5 * X_sim + np.random.normal(0, 0.5, n_sim)
        Y_sim_joint = 1.0 * X_sim + 0.6 * Z2_sim + 0.4 * Z3_sim + 0.3 * M_sim + np.random.normal(0, 0.5, n_sim)
        joint_mean = Y_sim_joint.mean()
        
        results['baseline'] = baseline_y
        results['single_x'] = single_x_mean
        results['single_z2'] = single_z2_mean
        results['joint'] = joint_mean
        results['expected_additive'] = baseline_y + (single_x_mean - baseline_y) + (single_z2_mean - baseline_y)
        
        return results
    
    def find_backdoor_adjustment_sets(self):
        """Automatically find valid backdoor adjustment sets"""
        # For simplicity, we'll identify adjustment sets for X -> Y
        target = ('X', 'Y')
        
        # Get all possible subsets of non-descendants of X (excluding X and Y)
        all_vars = set(self.dag.nodes()) - {'X', 'Y'}
        descendants_of_x = set(nx.descendants(self.dag, 'X'))
        candidates = all_vars - descendants_of_x  # Remove descendants to avoid collider bias
        
        valid_sets = []
        
        # Check all possible subsets
        for r in range(len(candidates) + 1):
            for subset in combinations(candidates, r):
                adjustment_set = set(subset)
                
                # Check backdoor criterion (simplified version)
                # Remove all edges from X to check if adjustment set blocks all backdoor paths
                dag_temp = self.dag.copy()
                out_edges = list(dag_temp.out_edges('X'))
                dag_temp.remove_edges_from(out_edges)
                
                # Check if adjustment set d-separates X and Y
                if self._blocks_all_paths(dag_temp, 'X', 'Y', adjustment_set):
                    valid_sets.append(list(subset))
        
        self.adjustment_sets = valid_sets
        return valid_sets
    
    def _blocks_all_paths(self, dag, source, target, adjustment_set):
        """Simplified d-separation check"""
        # This is a simplified version - full d-separation is more complex
        try:
            # If there's still a path after conditioning on adjustment_set, it's not valid
            paths = list(nx.all_simple_paths(dag, source, target, cutoff=10))
            for path in paths:
                if not any(node in adjustment_set for node in path[1:-1]):
                    return False
            return True
        except:
            return len(adjustment_set) > 0  # Simple fallback
    
    def compare_adjustment_methods(self):
        """Compare unadjusted, backdoor adjustment, and ground truth causal effects"""
        results = {}
        
        # 1. Unadjusted estimate (biased)
        lr_unadj = LinearRegression()
        lr_unadj.fit(self.data[['X']], self.data['Y'])
        unadjusted_effect = lr_unadj.coef_[0]
        
        # 2. Backdoor adjustment estimates
        adjustment_effects = {}
        for i, adj_set in enumerate(self.adjustment_sets[:3]):  # Test first 3 sets
            if len(adj_set) > 0:
                lr_adj = LinearRegression()
                features = ['X'] + adj_set
                lr_adj.fit(self.data[features], self.data['Y'])
                adjustment_effects[f'Set_{i}'] = lr_adj.coef_[0]  # Coefficient of X
            
        # 3. Ground truth effect (from data generation)
        ground_truth_effect = 1.0  # Direct effect of X on Y in our SCM
        
        results['unadjusted'] = unadjusted_effect
        results['adjusted'] = adjustment_effects
        results['ground_truth'] = ground_truth_effect
        results['adjustment_sets'] = self.adjustment_sets[:3]
        
        return results
    
    def demonstrate_collider_bias(self):
        """Create example showing 'incorrect adjustment' by conditioning on colliders"""
        # M is a collider (X -> M <- other factors through Y)
        # Conditioning on M should create bias
        
        # Correct adjustment (don't condition on M)
        lr_correct = LinearRegression()
        lr_correct.fit(self.data[['X', 'Z2', 'Z3']], self.data['Y'])  # Proper backdoor adjustment
        correct_effect = lr_correct.coef_[0]
        
        # Incorrect adjustment (condition on collider M)
        lr_incorrect = LinearRegression()
        lr_incorrect.fit(self.data[['X', 'Z2', 'Z3', 'M']], self.data['Y'])  # Including collider
        incorrect_effect = lr_incorrect.coef_[0]
        
        # Show the bias
        results = {
            'correct_adjustment': correct_effect,
            'collider_bias': incorrect_effect,
            'bias_magnitude': abs(incorrect_effect - correct_effect),
            'ground_truth': 1.0
        }
        
        return results
    
    def stratified_analysis_demo(self):
        """Demonstrate stratified analysis to show confounding control"""
        # Stratify by Z2 (major confounder) and show how effect changes
        results = {}
        
        # Overall effect (biased)
        lr_overall = LinearRegression()
        lr_overall.fit(self.data[['X']], self.data['Y'])
        overall_effect = lr_overall.coef_[0]
        
        # Stratified effects
        z2_tertiles = np.percentile(self.data['Z2'], [33, 67])
        strata_effects = []
        
        for i, (low, high) in enumerate([(self.data['Z2'].min(), z2_tertiles[0]),
                                       (z2_tertiles[0], z2_tertiles[1]),
                                       (z2_tertiles[1], self.data['Z2'].max())]):
            mask = (self.data['Z2'] >= low) & (self.data['Z2'] <= high)
            if mask.sum() > 20:  # Ensure sufficient data
                lr_strata = LinearRegression()
                lr_strata.fit(self.data.loc[mask, ['X']], self.data.loc[mask, 'Y'])
                strata_effects.append(lr_strata.coef_[0])
            else:
                strata_effects.append(np.nan)
        
        results['overall'] = overall_effect
        results['stratified'] = strata_effects
        results['average_stratified'] = np.nanmean(strata_effects)
        results['ground_truth'] = 1.0
        
        return results
    
    def propensity_matching_ate(self):
        """Estimate ATE using propensity score methods"""
        # Create binary treatment from X
        X_binary = (self.data['X'] > self.data['X'].median()).astype(int)
        
        # Estimate propensity scores
        from sklearn.linear_model import LogisticRegression
        ps_model = LogisticRegression()
        ps_model.fit(self.data[['Z1', 'Z2', 'Z3']], X_binary)
        propensity_scores = ps_model.predict_proba(self.data[['Z1', 'Z2', 'Z3']])[:, 1]
        
        # Simple inverse propensity weighting
        weights = np.where(X_binary == 1, 1/propensity_scores, 1/(1-propensity_scores))
        weights = np.clip(weights, 0.1, 10)  # Trim extreme weights
        
        # Weighted average treatment effect
        treated_outcomes = self.data.loc[X_binary == 1, 'Y']
        control_outcomes = self.data.loc[X_binary == 0, 'Y']
        treated_weights = weights[X_binary == 1]
        control_weights = weights[X_binary == 0]
        
        weighted_treated_mean = np.average(treated_outcomes, weights=treated_weights)
        weighted_control_mean = np.average(control_outcomes, weights=control_weights)
        ate_ipw = weighted_treated_mean - weighted_control_mean
        
        # Compare to simple difference in means (biased)
        simple_ate = treated_outcomes.mean() - control_outcomes.mean()
        
        results = {
            'ipw_ate': ate_ipw,
            'simple_ate': simple_ate,
            'propensity_mean': propensity_scores.mean(),
            'propensity_std': propensity_scores.std()
        }
        
        return results
    
    def run_complete_experiment(self):
        """Run complete experimental pipeline"""
        
        # 1. Generate DAG and data
        data, dag = self.generate_multivariate_dag()
        print(f"   Generated DAG with {len(dag.nodes())} nodes and {len(dag.edges())} edges")
        print(f"   Sample size: {len(data)} observations")
        
        # 2. Visualize DAG
        print("\n2. Visualizing DAG structure")
        self.visualize_dag()
        
        # 3. Intervention vs conditioning comparison
        print("\n3. Intervention vs conditioning at non-source nodes:")
        intervention_results = self.intervention_vs_conditioning()
        print("   X values:", intervention_results['x_values'])
        print("   P(Y|X) means:", [f"{x:.3f}" for x in intervention_results['observational'] if not np.isnan(x)])
        print("   P(Y|do(X)) means:", [f"{x:.3f}" for x in intervention_results['interventional']])
        
        # 4. Source node equivalence
        print("\n4. Source node equivalence demonstration:")
        source_results = self.demonstrate_source_node_equivalence()
        print("   Z1 values:", source_results['z_values'])
        print("   P(Y|Z1) means:", [f"{x:.3f}" for x in source_results['observational'] if not np.isnan(x)])
        print("   P(Y|do(Z1)) means:", [f"{x:.3f}" for x in source_results['interventional']])
        
        # 5. Multipoint intervention
        print("\n5. Multipoint intervention analysis:")
        multi_results = self.multipoint_intervention_analysis()
        print(f"   Baseline E[Y]: {multi_results['baseline']:.3f}")
        print(f"   E[Y|do(X=1)]: {multi_results['single_x']:.3f}")
        print(f"   E[Y|do(Z2=1)]: {multi_results['single_z2']:.3f}")
        print(f"   E[Y|do(X=1,Z2=1)]: {multi_results['joint']:.3f}")
        print(f"   Expected if additive: {multi_results['expected_additive']:.3f}")
        
        # 6. Backdoor adjustment sets
        print("\n6. Automatic backdoor adjustment set identification:")
        adjustment_sets = self.find_backdoor_adjustment_sets()
        print(f"   Found {len(adjustment_sets)} valid adjustment sets:")
        for i, adj_set in enumerate(adjustment_sets[:5]):
            print(f"   Set {i+1}: {adj_set if adj_set else 'Empty set'}")
        
        # 7. Adjustment method comparison
        print("\n7. Backdoor adjustment method comparison:")
        adj_results = self.compare_adjustment_methods()
        print(f"   Unadjusted estimate: {adj_results['unadjusted']:.3f}")
        print(f"   Ground truth effect: {adj_results['ground_truth']:.3f}")
        for method, effect in adj_results['adjusted'].items():
            print(f"   {method} adjustment: {effect:.3f}")
        
        # 8. Collider bias demonstration
        print("\n8. Collider bias demonstration:")
        collider_results = self.demonstrate_collider_bias()
        print(f"   Correct adjustment: {collider_results['correct_adjustment']:.3f}")
        print(f"   With collider bias: {collider_results['collider_bias']:.3f}")
        print(f"   Bias magnitude: {collider_results['bias_magnitude']:.3f}")
        
        # 9. Stratified analysis
        print("\n9. Stratified analysis for confounding control:")
        strata_results = self.stratified_analysis_demo()
        print(f"   Overall effect: {strata_results['overall']:.3f}")
        print(f"   Stratified effects: {[f'{x:.3f}' for x in strata_results['stratified'] if not np.isnan(x)]}")
        print(f"   Average stratified: {strata_results['average_stratified']:.3f}")
        
        # 10. Propensity score methods
        print("\n10. Propensity score matching/weighting:")
        ps_results = self.propensity_matching_ate()
        print(f"    Simple ATE: {ps_results['simple_ate']:.3f}")
        print(f"    IPW ATE: {ps_results['ipw_ate']:.3f}")
        print(f"    Propensity score mean: {ps_results['propensity_mean']:.3f}")
        
        # Store all results
        self.experiment_results = {
            'intervention': intervention_results,
            'source_equivalence': source_results,
            'multipoint': multi_results,
            'adjustment_sets': adjustment_sets,
            'adjustment_comparison': adj_results,
            'collider_bias': collider_results,
            'stratified': strata_results,
            'propensity': ps_results
        }

# Run experiment
if __name__ == "__main__":
    experiment = MultivariateInterventionPractice()
    experiment.run_complete_experiment()
