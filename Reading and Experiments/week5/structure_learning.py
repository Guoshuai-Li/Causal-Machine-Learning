import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations, permutations

np.random.seed(42)

class CausalStructureLearning:
    def __init__(self):
        self.scaler = StandardScaler()
        self.true_dag = None
        self.learned_graphs = {}
        self.data = None
        
    def generate_random_dag(self, n_vars=5, edge_prob=0.3, n_samples=1000):
        """Generate random DAG with known structure for evaluation"""
        var_names = [f'X{i}' for i in range(n_vars)]
        
        # Generate random DAG structure
        self.true_dag = nx.DiGraph()
        self.true_dag.add_nodes_from(var_names)
        
        # Add edges respecting topological order
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                if np.random.random() < edge_prob:
                    self.true_dag.add_edge(f'X{i}', f'X{j}')
        
        # Generate data following the DAG structure
        data_dict = {}
        
        # Topological sort to ensure causal order
        topo_order = list(nx.topological_sort(self.true_dag))
        
        for var in topo_order:
            parents = list(self.true_dag.predecessors(var))
            
            if len(parents) == 0:
                # Root node - generate from noise
                data_dict[var] = np.random.normal(0, 1, n_samples)
            else:
                # Child node - linear combination of parents + noise
                linear_combination = np.zeros(n_samples)
                for parent in parents:
                    # Random coefficient between 0.5 and 2.0
                    coef = np.random.uniform(0.5, 2.0)
                    linear_combination += coef * data_dict[parent]
                
                # Add noise
                noise = np.random.normal(0, 0.5, n_samples)
                data_dict[var] = linear_combination + noise
        
        self.data = pd.DataFrame(data_dict)
        return self.data, self.true_dag
    
    def pc_algorithm_simplified(self, alpha=0.05):
        """Simplified PC algorithm implementation with conditional independence testing"""
        variables = list(self.data.columns)
        n_vars = len(variables)
        
        # Step 1: Start with complete undirected graph
        undirected_edges = set()
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                undirected_edges.add((variables[i], variables[j]))
        
        # Step 2: Remove edges based on conditional independence tests
        # Order 0: Test marginal independence
        edges_to_remove = set()
        for var1, var2 in undirected_edges:
            if self._independence_test(var1, var2, [], alpha):
                edges_to_remove.add((var1, var2))
        
        undirected_edges -= edges_to_remove
        
        # Order 1: Test conditional independence given one variable
        edges_to_remove = set()
        for var1, var2 in undirected_edges:
            for conditioning_var in variables:
                if conditioning_var != var1 and conditioning_var != var2:
                    if self._independence_test(var1, var2, [conditioning_var], alpha):
                        edges_to_remove.add((var1, var2))
                        break
        
        undirected_edges -= edges_to_remove
        
        # Step 3: Orient edges (simplified version)
        directed_edges = self._orient_edges_simplified(undirected_edges, variables)
        
        # Create learned graph
        learned_graph = nx.DiGraph()
        learned_graph.add_nodes_from(variables)
        learned_graph.add_edges_from(directed_edges)
        
        self.learned_graphs['PC'] = learned_graph
        return learned_graph
    
    def _independence_test(self, var1, var2, conditioning_vars, alpha):
        """Test conditional independence using partial correlation"""
        if len(conditioning_vars) == 0:
            # Marginal correlation test
            corr, p_value = stats.pearsonr(self.data[var1], self.data[var2])
            return p_value > alpha
        else:
            # Partial correlation test
            try:
                # Create feature matrix
                all_vars = [var1, var2] + conditioning_vars
                data_subset = self.data[all_vars].dropna()
                
                if len(data_subset) < 10:  # Too few samples
                    return False
                
                # Regress var1 on conditioning variables
                lr1 = LinearRegression()
                lr1.fit(data_subset[conditioning_vars], data_subset[var1])
                residual1 = data_subset[var1] - lr1.predict(data_subset[conditioning_vars])
                
                # Regress var2 on conditioning variables
                lr2 = LinearRegression()
                lr2.fit(data_subset[conditioning_vars], data_subset[var2])
                residual2 = data_subset[var2] - lr2.predict(data_subset[conditioning_vars])
                
                # Test correlation of residuals
                if len(residual1) > 3:
                    corr, p_value = stats.pearsonr(residual1, residual2)
                    return p_value > alpha
                else:
                    return False
            except:
                return False
    
    def _orient_edges_simplified(self, undirected_edges, variables):
        """Simplified edge orientation using v-structures and topological constraints"""
        directed_edges = []
        
        # Convert undirected edges to adjacency info
        adjacency = {var: set() for var in variables}
        for var1, var2 in undirected_edges:
            adjacency[var1].add(var2)
            adjacency[var2].add(var1)
        
        # Simple heuristic: orient based on marginal variances
        # Variables with higher variance tend to be causes (rough heuristic)
        variances = {var: self.data[var].var() for var in variables}
        
        for var1, var2 in undirected_edges:
            if variances[var1] > variances[var2]:
                directed_edges.append((var1, var2))
            else:
                directed_edges.append((var2, var1))
        
        return directed_edges
    
    def ges_algorithm_simplified(self):
        """Simplified GES (Greedy Equivalence Search) using BIC scoring"""
        variables = list(self.data.columns)
        n_vars = len(variables)
        
        # Start with empty graph
        current_graph = nx.DiGraph()
        current_graph.add_nodes_from(variables)
        current_score = self._calculate_bic_score(current_graph)
        
        # Forward phase: Add edges
        improved = True
        while improved:
            improved = False
            best_score = current_score
            best_edge = None
            
            # Try adding each possible edge
            for var1, var2 in combinations(variables, 2):
                for direction in [(var1, var2), (var2, var1)]:
                    if not current_graph.has_edge(*direction):
                        # Try adding this edge
                        test_graph = current_graph.copy()
                        test_graph.add_edge(*direction)
                        
                        # Check if still acyclic
                        if nx.is_directed_acyclic_graph(test_graph):
                            score = self._calculate_bic_score(test_graph)
                            if score > best_score:
                                best_score = score
                                best_edge = direction
                                improved = True
            
            if improved:
                current_graph.add_edge(*best_edge)
                current_score = best_score
        
        # Backward phase: Remove edges (simplified)
        improved = True
        while improved:
            improved = False
            best_score = current_score
            best_edge = None
            
            # Try removing each edge
            for edge in list(current_graph.edges()):
                test_graph = current_graph.copy()
                test_graph.remove_edge(*edge)
                score = self._calculate_bic_score(test_graph)
                if score > best_score:
                    best_score = score
                    best_edge = edge
                    improved = True
            
            if improved:
                current_graph.remove_edge(*best_edge)
                current_score = best_score
        
        self.learned_graphs['GES'] = current_graph
        return current_graph
    
    def _calculate_bic_score(self, graph):
        """Calculate BIC score for a given graph structure"""
        total_score = 0
        n = len(self.data)
        
        for node in graph.nodes():
            parents = list(graph.predecessors(node))
            
            if len(parents) == 0:
                # No parents - just variance of the node
                variance = self.data[node].var()
                log_likelihood = -0.5 * n * np.log(2 * np.pi * variance) - 0.5 * n
                penalty = 0.5 * np.log(n)  # One parameter (variance)
            else:
                # Has parents - regression model
                try:
                    lr = LinearRegression()
                    lr.fit(self.data[parents], self.data[node])
                    predictions = lr.predict(self.data[parents])
                    residuals = self.data[node] - predictions
                    mse = np.mean(residuals**2)
                    
                    if mse > 0:
                        log_likelihood = -0.5 * n * np.log(2 * np.pi * mse) - 0.5 * n
                        penalty = 0.5 * (len(parents) + 1) * np.log(n)  # Parameters: coefficients + variance
                    else:
                        log_likelihood = 0
                        penalty = 0.5 * (len(parents) + 1) * np.log(n)
                except:
                    log_likelihood = -1e6  # Very bad score
                    penalty = 0
            
            total_score += log_likelihood - penalty
        
        return total_score
    
    def evaluate_structure_recovery(self):
        """Evaluate how well learned structures match the true DAG"""
        results = {}
        
        true_edges = set(self.true_dag.edges())
        
        for method_name, learned_graph in self.learned_graphs.items():
            learned_edges = set(learned_graph.edges())
            
            # Calculate metrics
            tp = len(true_edges & learned_edges)  # True positives
            fp = len(learned_edges - true_edges)  # False positives
            fn = len(true_edges - learned_edges)  # False negatives
            
            # Avoid division by zero
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Structural Hamming Distance (SHD)
            shd = fp + fn
            
            results[method_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'shd': shd,
                'true_edges': len(true_edges),
                'learned_edges': len(learned_edges),
                'correct_edges': tp
            }
        
        return results
    
    def demonstrate_markov_equivalence(self):
        """Demonstrate Markov equivalence with multiple DAGs having same conditional independencies"""
        # Create three equivalent DAGs
        vars = ['A', 'B', 'C']
        
        # DAG 1: A -> B -> C
        dag1 = nx.DiGraph()
        dag1.add_nodes_from(vars)
        dag1.add_edges_from([('A', 'B'), ('B', 'C')])
        
        # DAG 2: A <- B -> C  
        dag2 = nx.DiGraph()
        dag2.add_nodes_from(vars)
        dag2.add_edges_from([('B', 'A'), ('B', 'C')])
        
        # DAG 3: A -> B <- C
        dag3 = nx.DiGraph()
        dag3.add_nodes_from(vars)
        dag3.add_edges_from([('A', 'B'), ('C', 'B')])
        
        equivalent_dags = {'Chain_A->B->C': dag1, 'Fork_A<-B->C': dag2, 'Collider_A->B<-C': dag3}
        
        # Generate data for each DAG and test conditional independencies
        equivalence_results = {}
        
        for dag_name, dag in equivalent_dags.items():
            # Generate small dataset for this DAG
            data_dict = {}
            n_samples = 500
            
            topo_order = list(nx.topological_sort(dag))
            
            for var in topo_order:
                parents = list(dag.predecessors(var))
                
                if len(parents) == 0:
                    data_dict[var] = np.random.normal(0, 1, n_samples)
                else:
                    linear_combination = np.zeros(n_samples)
                    for parent in parents:
                        coef = 1.0  # Fixed coefficient for consistency
                        linear_combination += coef * data_dict[parent]
                    
                    noise = np.random.normal(0, 0.5, n_samples)
                    data_dict[var] = linear_combination + noise
            
            dag_data = pd.DataFrame(data_dict)
            
            # Test key conditional independence: A ⊥ C | B
            temp_data = self.data
            self.data = dag_data
            is_independent = self._independence_test('A', 'C', ['B'], alpha=0.05)
            self.data = temp_data
            
            equivalence_results[dag_name] = {
                'structure': list(dag.edges()),
                'A_indep_C_given_B': is_independent,
                'data_shape': dag_data.shape
            }
        
        return equivalent_dags, equivalence_results
    
    def visualize_cpdag(self, learned_graph=None):
        """Visualize CPDAG (Completed Partially Directed Acyclic Graph)"""
        if learned_graph is None:
            learned_graph = self.learned_graphs.get('PC', nx.DiGraph())
        
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(learned_graph, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(learned_graph, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.8)
        
        # Draw directed edges (solid arrows)
        directed_edges = list(learned_graph.edges())
        nx.draw_networkx_edges(learned_graph, pos, edgelist=directed_edges,
                              edge_color='blue', arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(learned_graph, pos, font_size=10, font_weight='bold')
        
        plt.title('Learned Causal Structure (CPDAG)', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def demonstrate_faithfulness_violation(self):
        """Create dataset that violates faithfulness assumption"""
        n_samples = 1000
        
        # Create a scenario where X -> Z <- Y but X ⊥ Y marginally due to parameter cancellation
        X = np.random.normal(0, 1, n_samples)
        Y = np.random.normal(0, 1, n_samples)
        
        # Z depends on both X and Y with equal but opposite effects
        # This creates a situation where X and Y appear independent marginally
        # but are dependent given Z (violating faithfulness)
        Z = 1.0 * X - 1.0 * Y + np.random.normal(0, 0.1, n_samples)
        
        faithfulness_data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
        
        # Test independencies
        temp_data = self.data
        self.data = faithfulness_data
        
        # Test marginal independence X ⊥ Y (should be true due to parameter cancellation)
        marginal_indep = self._independence_test('X', 'Y', [], alpha=0.05)
        
        # Test conditional independence X ⊥ Y | Z (should be false)
        conditional_indep = self._independence_test('X', 'Y', ['Z'], alpha=0.05)
        
        self.data = temp_data
        
        faithfulness_results = {
            'marginal_independence_XY': marginal_indep,
            'conditional_independence_XY_given_Z': conditional_indep,
            'violates_faithfulness': marginal_indep and not conditional_indep,
            'data_correlations': {
                'corr_XY': faithfulness_data[['X', 'Y']].corr().iloc[0, 1],
                'corr_XZ': faithfulness_data[['X', 'Z']].corr().iloc[0, 1],
                'corr_YZ': faithfulness_data[['Y', 'Z']].corr().iloc[0, 1]
            }
        }
        
        return faithfulness_data, faithfulness_results
    
    def compare_sample_size_effects(self):
        """Compare structure learning performance across different sample sizes"""
        sample_sizes = [100, 500, 1000, 2000]
        performance_results = {}
        
        # Store original data
        original_data = self.data.copy()
        
        for n in sample_sizes:
            # Subsample data
            if n <= len(original_data):
                sampled_data = original_data.sample(n=n, random_state=42)
            else:
                sampled_data = original_data
            
            self.data = sampled_data
            
            # Run PC algorithm
            try:
                learned_pc = self.pc_algorithm_simplified(alpha=0.05)
                self.learned_graphs = {'PC': learned_pc}
                
                # Evaluate performance
                eval_results = self.evaluate_structure_recovery()
                performance_results[n] = eval_results['PC']
            except:
                performance_results[n] = {
                    'precision': 0, 'recall': 0, 'f1_score': 0, 'shd': 999
                }
        
        # Restore original data
        self.data = original_data
        
        return performance_results
    
    def run_complete_experiment(self):
        """Run complete structure learning experimental pipeline"""
        print("=== Week 5: Causal Structure Learning ===\n")
        
        # 1. Generate random DAG and data
        print("1. Generating random causal DAG and observational data...")
        data, true_dag = self.generate_random_dag(n_vars=5, edge_prob=0.4, n_samples=1000)
        print(f"   Generated DAG with {len(true_dag.nodes())} variables and {len(true_dag.edges())} edges")
        print(f"   True edges: {list(true_dag.edges())}")
        print(f"   Sample size: {len(data)} observations")
        
        # 2. PC Algorithm
        print("\n2. PC Algorithm for conditional independence-based learning:")
        pc_graph = self.pc_algorithm_simplified(alpha=0.05)
        print(f"   PC learned edges: {list(pc_graph.edges())}")
        print(f"   PC graph has {len(pc_graph.edges())} edges")
        
        # 3. GES Algorithm  
        print("\n3. GES Algorithm for score-based learning:")
        ges_graph = self.ges_algorithm_simplified()
        print(f"   GES learned edges: {list(ges_graph.edges())}")
        print(f"   GES graph has {len(ges_graph.edges())} edges")
        
        # 4. Structure recovery evaluation
        print("\n4. Structure recovery evaluation:")
        eval_results = self.evaluate_structure_recovery()
        
        for method, metrics in eval_results.items():
            print(f"   {method} Algorithm:")
            print(f"   Precision: {metrics['precision']:.3f}")
            print(f"   Recall: {metrics['recall']:.3f}")
            print(f"   F1-Score: {metrics['f1_score']:.3f}")
            print(f"   SHD (Structural Hamming Distance): {metrics['shd']}")
            print(f"   Correct edges: {metrics['correct_edges']}/{metrics['true_edges']}")
        
        # 5. Markov equivalence demonstration
        print("\n5. Markov equivalence class demonstration:")
        equiv_dags, equiv_results = self.demonstrate_markov_equivalence()
        
        print("   Testing A ⊥ C | B for different equivalent structures:")
        for dag_name, results in equiv_results.items():
            print(f"   {dag_name}: A⊥C|B = {results['A_indep_C_given_B']}")
            print(f"   Structure: {results['structure']}")
        
        # 6. CPDAG visualization
        print("\n6. Visualizing learned causal structure (CPDAG)...")
        self.visualize_cpdag(pc_graph)
        
        # 7. Faithfulness violation
        print("\n7. Faithfulness assumption violation demonstration:")
        faith_data, faith_results = self.demonstrate_faithfulness_violation()
        print(f"   Marginal independence X⊥Y: {faith_results['marginal_independence_XY']}")
        print(f"   Conditional independence X⊥Y|Z: {faith_results['conditional_independence_XY_given_Z']}")
        print(f"   Violates faithfulness: {faith_results['violates_faithfulness']}")
        print("   Correlations:")
        for corr_name, corr_val in faith_results['data_correlations'].items():
            print(f"   {corr_name}: {corr_val:.3f}")
        
        # 8. Sample size effects
        print("\n8. Sample size effects on structure learning:")
        sample_effects = self.compare_sample_size_effects()
        
        print("   Performance by sample size:")
        for n, metrics in sample_effects.items():
            print(f"   n={n}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}, SHD={metrics['shd']}")
        
        # Store results for further analysis
        self.experiment_results = {
            'true_structure': true_dag,
            'learned_structures': self.learned_graphs,
            'evaluation_metrics': eval_results,
            'markov_equivalence': equiv_results,
            'faithfulness_test': faith_results,
            'sample_size_effects': sample_effects
        }

# Run experiment
if __name__ == "__main__":
    experiment = CausalStructureLearning()
    experiment.run_complete_experiment()
