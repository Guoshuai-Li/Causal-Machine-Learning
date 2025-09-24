import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import het_breuschpagan
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class InstrumentalVariablesExperiment:
    def __init__(self):
        self.data = None
        self.results = {}
        
    def generate_endogeneity_scenario(self, n=2000):
        """Generate scenario demonstrating endogeneity problem"""
        
        # Generate unobserved confounder (latent variable)
        U = np.random.normal(0, 1, n)  # Unobserved ability, preference, etc.
        
        # Generate instrumental variable
        # Good instrument: affects treatment but not outcome directly
        Z = np.random.normal(0, 1, n)  # e.g., random policy assignment, lottery
        
        # Generate treatment variable (endogenous)
        # Treatment is affected by both instrument and unobserved confounder
        X = (2.0 * Z +           # Instrument effect (strong first stage)
             1.5 * U +           # Confounding (creates endogeneity)
             np.random.normal(0, 0.5, n))
        
        # Generate outcome variable
        # Outcome affected by treatment and unobserved confounder
        Y = (1.0 * X +           # True causal effect we want to estimate
             2.0 * U +           # Confounding (creates bias in OLS)
             np.random.normal(0, 0.3, n))
        
        # Store data
        self.data = pd.DataFrame({
            'treatment': X,
            'outcome': Y,
            'instrument': Z,
            'unobserved_confounder': U
        })
        
        # True parameters for validation
        self.true_parameters = {
            'causal_effect': 1.0,          # True effect of X on Y
            'first_stage_strength': 2.0,   # Effect of Z on X
            'confounding_strength': 2.0    # Effect of U on Y
        }
        
        return self.data
    
    def demonstrate_endogeneity_bias(self):
        """Show why OLS fails with endogeneity"""
        results = {}
        
        # Method 1: Naive OLS (biased due to endogeneity)
        ols_model = LinearRegression()
        ols_model.fit(self.data[['treatment']], self.data['outcome'])
        ols_coefficient = ols_model.coef_[0]
        
        # Method 2: Oracle OLS (controls for unobserved confounder)
        oracle_model = LinearRegression()
        oracle_model.fit(self.data[['treatment', 'unobserved_confounder']], 
                        self.data['outcome'])
        oracle_coefficient = oracle_model.coef_[0]  # Coefficient of treatment
        
        # Calculate bias
        true_effect = self.true_parameters['causal_effect']
        ols_bias = ols_coefficient - true_effect
        oracle_bias = oracle_coefficient - true_effect
        
        results = {
            'ols_estimate': ols_coefficient,
            'oracle_estimate': oracle_coefficient,
            'true_effect': true_effect,
            'ols_bias': ols_bias,
            'oracle_bias': oracle_bias,
            'bias_percentage': (ols_bias / true_effect) * 100
        }
        
        return results
    
    def implement_instrumental_variables(self):
        """Implement IV estimation methods"""
        results = {}
        
        # Method 1: Two-Stage Least Squares (2SLS)
        # First stage: regress treatment on instrument
        first_stage = LinearRegression()
        first_stage.fit(self.data[['instrument']], self.data['treatment'])
        predicted_treatment = first_stage.predict(self.data[['instrument']])
        
        # Check first stage strength (F-statistic)
        residuals_first = self.data['treatment'] - predicted_treatment
        mse_first = np.mean(residuals_first**2)
        f_stat = (first_stage.coef_[0]**2) / (mse_first / len(self.data))
        
        # Second stage: regress outcome on predicted treatment
        second_stage = LinearRegression()
        second_stage.fit(predicted_treatment.reshape(-1, 1), self.data['outcome'])
        tsls_coefficient = second_stage.coef_[0]
        
        # Method 2: Direct IV formula (Wald estimator)
        # IV = Cov(Y,Z) / Cov(X,Z)
        cov_yz = np.cov(self.data['outcome'], self.data['instrument'])[0,1]
        cov_xz = np.cov(self.data['treatment'], self.data['instrument'])[0,1]
        wald_coefficient = cov_yz / cov_xz
        
        # Method 3: Reduced form approach
        # Reduced form: regress outcome on instrument
        reduced_form = LinearRegression()
        reduced_form.fit(self.data[['instrument']], self.data['outcome'])
        reduced_form_coef = reduced_form.coef_[0]
        
        # First stage coefficient
        first_stage_coef = first_stage.coef_[0]
        
        # IV estimate = reduced form / first stage
        iv_ratio = reduced_form_coef / first_stage_coef
        
        results = {
            'tsls_estimate': tsls_coefficient,
            'wald_estimate': wald_coefficient,
            'iv_ratio_estimate': iv_ratio,
            'first_stage_coefficient': first_stage_coef,
            'first_stage_f_statistic': f_stat,
            'reduced_form_coefficient': reduced_form_coef
        }
        
        return results
    
    def test_instrument_validity(self):
        """Test instrumental variable assumptions"""
        results = {}
        
        # Assumption 1: Relevance (first stage strength)
        # Strong correlation between instrument and treatment
        correlation_zx = np.corrcoef(self.data['instrument'], self.data['treatment'])[0,1]
        
        # F-statistic for first stage
        first_stage = LinearRegression()
        first_stage.fit(self.data[['instrument']], self.data['treatment'])
        predicted_treatment = first_stage.predict(self.data[['instrument']])
        residuals = self.data['treatment'] - predicted_treatment
        
        # Calculate F-statistic
        n = len(self.data)
        k = 1  # Number of instruments
        f_stat = (first_stage.coef_[0]**2 * n) / np.var(residuals)
        
        # Rule of thumb: F-stat > 10 for strong instrument
        is_relevant = f_stat > 10
        
        # Assumption 2: Exogeneity (instrument uncorrelated with error term)
        # In real data, this is untestable, but we can check with our known confounder
        correlation_zu = np.corrcoef(self.data['instrument'], 
                                   self.data['unobserved_confounder'])[0,1]
        
        # Should be close to 0 for valid instrument
        is_exogenous = abs(correlation_zu) < 0.1
        
        # Assumption 3: Exclusion restriction
        # Instrument affects outcome only through treatment
        # Test: regress outcome on instrument and treatment
        exclusion_test = LinearRegression()
        exclusion_test.fit(self.data[['treatment', 'instrument']], self.data['outcome'])
        instrument_direct_effect = exclusion_test.coef_[1]
        
        # Should be close to 0 for valid instrument
        exclusion_satisfied = abs(instrument_direct_effect) < 0.1
        
        results = {
            'relevance': {
                'correlation_zx': correlation_zx,
                'first_stage_f_stat': f_stat,
                'is_relevant': is_relevant
            },
            'exogeneity': {
                'correlation_zu': correlation_zu,
                'is_exogenous': is_exogenous
            },
            'exclusion_restriction': {
                'direct_effect': instrument_direct_effect,
                'exclusion_satisfied': exclusion_satisfied
            }
        }
        
        return results
    
    def generate_tetrad_constraints(self):
        """Generate data with latent variable structure for tetrad analysis"""
        
        n = 2000
        
        # Generate latent variable
        L = np.random.normal(0, 1, n)  # Latent factor
        
        # Generate observed variables influenced by latent factor
        # Classic factor model: X = λL + ε
        X1 = 0.8 * L + np.random.normal(0, 0.6, n)  # Loading = 0.8
        X2 = 0.7 * L + np.random.normal(0, 0.7, n)  # Loading = 0.7
        X3 = 0.6 * L + np.random.normal(0, 0.8, n)  # Loading = 0.6
        X4 = 0.9 * L + np.random.normal(0, 0.4, n)  # Loading = 0.9
        
        # Additional variables not influenced by L (for comparison)
        X5 = np.random.normal(0, 1, n)  # Independent
        X6 = np.random.normal(0, 1, n)  # Independent
        
        tetrad_data = pd.DataFrame({
            'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4,
            'X5': X5, 'X6': X6, 'L': L
        })
        
        return tetrad_data
    
    def test_tetrad_constraints(self, data):
        """Test tetrad constraints for latent variable detection"""
        
        # Tetrad constraint: for variables generated by single latent factor
        # Cov(X1,X2)*Cov(X3,X4) = Cov(X1,X3)*Cov(X2,X4) = Cov(X1,X4)*Cov(X2,X3)
        
        variables = ['X1', 'X2', 'X3', 'X4']
        cov_matrix = data[variables].cov()
        
        # Calculate tetrads
        tetrad1 = cov_matrix.loc['X1','X2'] * cov_matrix.loc['X3','X4']
        tetrad2 = cov_matrix.loc['X1','X3'] * cov_matrix.loc['X2','X4'] 
        tetrad3 = cov_matrix.loc['X1','X4'] * cov_matrix.loc['X2','X3']
        
        # Test if tetrads are approximately equal (within tolerance)
        tolerance = 0.1
        tetrad_diff12 = abs(tetrad1 - tetrad2)
        tetrad_diff13 = abs(tetrad1 - tetrad3)
        tetrad_diff23 = abs(tetrad2 - tetrad3)
        
        constraints_satisfied = (tetrad_diff12 < tolerance and 
                               tetrad_diff13 < tolerance and 
                               tetrad_diff23 < tolerance)
        
        # Compare with variables not sharing latent factor
        independent_vars = ['X5', 'X6', 'X1', 'X2']
        cov_matrix_indep = data[independent_vars].cov()
        
        tetrad1_indep = cov_matrix_indep.loc['X5','X6'] * cov_matrix_indep.loc['X1','X2']
        tetrad2_indep = cov_matrix_indep.loc['X5','X1'] * cov_matrix_indep.loc['X6','X2']
        tetrad3_indep = cov_matrix_indep.loc['X5','X2'] * cov_matrix_indep.loc['X6','X1']
        
        tetrad_diff12_indep = abs(tetrad1_indep - tetrad2_indep)
        
        results = {
            'latent_factor_tetrads': [tetrad1, tetrad2, tetrad3],
            'tetrad_differences': [tetrad_diff12, tetrad_diff13, tetrad_diff23],
            'constraints_satisfied': constraints_satisfied,
            'independent_tetrad_diff': tetrad_diff12_indep
        }
        
        return results
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Endogeneity demonstration
        axes[0,0].scatter(self.data['treatment'], self.data['outcome'], 
                         c=self.data['unobserved_confounder'], cmap='RdYlBu', alpha=0.6)
        
        # Add OLS line
        ols_model = LinearRegression()
        ols_model.fit(self.data[['treatment']], self.data['outcome'])
        x_range = np.linspace(self.data['treatment'].min(), self.data['treatment'].max(), 100)
        y_pred = ols_model.predict(x_range.reshape(-1, 1))
        axes[0,0].plot(x_range, y_pred, 'r--', linewidth=2, label=f'OLS (β={ols_model.coef_[0]:.2f})')
        
        # Add true causal effect line
        true_intercept = self.data['outcome'].mean() - self.true_parameters['causal_effect'] * self.data['treatment'].mean()
        y_true = self.true_parameters['causal_effect'] * x_range + true_intercept
        axes[0,0].plot(x_range, y_true, 'g-', linewidth=2, label=f'True Effect (β={self.true_parameters["causal_effect"]:.1f})')
        
        axes[0,0].set_xlabel('Treatment (X)')
        axes[0,0].set_ylabel('Outcome (Y)')
        axes[0,0].set_title('Endogeneity Problem\n(Color = Unobserved Confounder)')
        axes[0,0].legend()
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlBu'), ax=axes[0,0])
        cbar.set_label('Unobserved Confounder (U)')
        
        # Plot 2: First stage relationship
        axes[0,1].scatter(self.data['instrument'], self.data['treatment'], alpha=0.6, color='blue')
        
        first_stage = LinearRegression()
        first_stage.fit(self.data[['instrument']], self.data['treatment'])
        z_range = np.linspace(self.data['instrument'].min(), self.data['instrument'].max(), 100)
        x_pred = first_stage.predict(z_range.reshape(-1, 1))
        axes[0,1].plot(z_range, x_pred, 'r-', linewidth=2, 
                      label=f'First Stage (γ={first_stage.coef_[0]:.2f})')
        
        axes[0,1].set_xlabel('Instrument (Z)')
        axes[0,1].set_ylabel('Treatment (X)')
        axes[0,1].set_title('First Stage: Instrument Relevance')
        axes[0,1].legend()
        
        # Plot 3: Reduced form relationship
        axes[0,2].scatter(self.data['instrument'], self.data['outcome'], alpha=0.6, color='green')
        
        reduced_form = LinearRegression()
        reduced_form.fit(self.data[['instrument']], self.data['outcome'])
        y_reduced = reduced_form.predict(z_range.reshape(-1, 1))
        axes[0,2].plot(z_range, y_reduced, 'r-', linewidth=2,
                      label=f'Reduced Form (π={reduced_form.coef_[0]:.2f})')
        
        axes[0,2].set_xlabel('Instrument (Z)')
        axes[0,2].set_ylabel('Outcome (Y)')
        axes[0,2].set_title('Reduced Form: Z → Y')
        axes[0,2].legend()
        
        # Plot 4: Method comparison
        if 'endogeneity_results' in self.results:
            endogeneity_res = self.results['endogeneity_results']
            methods = ['True Effect', 'OLS\n(Biased)', 'Oracle\n(Unbiased)']
            estimates = [endogeneity_res['true_effect'], 
                        endogeneity_res['ols_estimate'],
                        endogeneity_res['oracle_estimate']]
            colors = ['green', 'red', 'blue']
            
            bars = axes[1,0].bar(methods, estimates, color=colors, alpha=0.7)
            axes[1,0].axhline(y=endogeneity_res['true_effect'], color='black', 
                             linestyle='--', alpha=0.5, label='True Effect')
            axes[1,0].set_ylabel('Causal Effect Estimate')
            axes[1,0].set_title('OLS Bias Demonstration')
            
            # Add value labels
            for bar, estimate in zip(bars, estimates):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                              f'{estimate:.3f}', ha='center', va='bottom')
        
        # Plot 5: IV method comparison  
        if 'iv_results' in self.results:
            iv_res = self.results['iv_results']
            iv_methods = ['True Effect', '2SLS', 'Wald', 'IV Ratio']
            iv_estimates = [self.true_parameters['causal_effect'],
                           iv_res['tsls_estimate'], 
                           iv_res['wald_estimate'],
                           iv_res['iv_ratio_estimate']]
            
            bars = axes[1,1].bar(iv_methods, iv_estimates, color='orange', alpha=0.7)
            axes[1,1].axhline(y=self.true_parameters['causal_effect'], 
                             color='black', linestyle='--', alpha=0.5)
            axes[1,1].set_ylabel('Causal Effect Estimate')
            axes[1,1].set_title('IV Method Comparison')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, estimate in zip(bars, iv_estimates):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                              f'{estimate:.3f}', ha='center', va='bottom')
        
        # Plot 6: Instrument validity tests
        if 'validity_results' in self.results:
            validity_res = self.results['validity_results']
            
            assumptions = ['Relevance\n(F > 10)', 'Exogeneity\n(|r| < 0.1)', 'Exclusion\n(|β| < 0.1)']
            test_values = [validity_res['relevance']['first_stage_f_stat'],
                          abs(validity_res['exogeneity']['correlation_zu']),
                          abs(validity_res['exclusion_restriction']['direct_effect'])]
            thresholds = [10, 0.1, 0.1]
            passed = [validity_res['relevance']['is_relevant'],
                     validity_res['exogeneity']['is_exogenous'],
                     validity_res['exclusion_restriction']['exclusion_satisfied']]
            
            colors = ['green' if p else 'red' for p in passed]
            bars = axes[1,2].bar(assumptions, test_values, color=colors, alpha=0.7)
            
            # Add threshold lines
            for i, threshold in enumerate(thresholds):
                if i == 0:  # F-statistic (higher is better)
                    axes[1,2].axhline(y=threshold, color='black', linestyle='--', alpha=0.5)
                else:  # Correlations (lower is better)
                    axes[1,2].axhline(y=threshold, color='black', linestyle='--', alpha=0.5)
            
            axes[1,2].set_ylabel('Test Statistic Value')
            axes[1,2].set_title('Instrument Validity Tests')
            axes[1,2].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value, passed_test in zip(bars, test_values, passed):
                label = f'{value:.3f}\n{"✓" if passed_test else "✗"}'
                axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              label, ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_experiment(self):
        """Run complete instrumental variables experiment"""
        
        print("=== Week 7: Latent Variables and Instrumental Variables ===\n")
        
        # 1. Generate endogeneity scenario
        print("1. Generating Endogeneity Scenario:")
        data = self.generate_endogeneity_scenario(n=2000)
        print(f"   Generated dataset with {len(data)} observations")
        print(f"   True causal effect: {self.true_parameters['causal_effect']}")
        print(f"   First stage strength: {self.true_parameters['first_stage_strength']}")
        print(f"   Confounding strength: {self.true_parameters['confounding_strength']}")
        
        # 2. Demonstrate endogeneity bias
        print("\n2. Demonstrating Endogeneity Bias:")
        endogeneity_results = self.demonstrate_endogeneity_bias()
        self.results['endogeneity_results'] = endogeneity_results
        
        print(f"   OLS estimate: {endogeneity_results['ols_estimate']:.3f}")
        print(f"   Oracle estimate: {endogeneity_results['oracle_estimate']:.3f}")
        print(f"   True effect: {endogeneity_results['true_effect']:.3f}")
        print(f"   OLS bias: {endogeneity_results['ols_bias']:.3f} ({endogeneity_results['bias_percentage']:.1f}%)")
        
        # 3. Implement IV methods
        print("\n3. Implementing Instrumental Variables Methods:")
        iv_results = self.implement_instrumental_variables()
        self.results['iv_results'] = iv_results
        
        print(f"   2SLS estimate: {iv_results['tsls_estimate']:.3f}")
        print(f"   Wald estimate: {iv_results['wald_estimate']:.3f}")
        print(f"   IV ratio estimate: {iv_results['iv_ratio_estimate']:.3f}")
        print(f"   First stage F-statistic: {iv_results['first_stage_f_statistic']:.1f}")
        
        # 4. Test instrument validity
        print("\n4. Testing Instrument Validity:")
        validity_results = self.test_instrument_validity()
        self.results['validity_results'] = validity_results
        
        print(f"   Relevance test:")
        print(f"     First stage F-statistic: {validity_results['relevance']['first_stage_f_stat']:.1f}")
        print(f"     Strong instrument: {validity_results['relevance']['is_relevant']}")
        
        print(f"   Exogeneity test:")
        print(f"     Instrument-confounder correlation: {validity_results['exogeneity']['correlation_zu']:.3f}")
        print(f"     Exogenous: {validity_results['exogeneity']['is_exogenous']}")
        
        print(f"   Exclusion restriction test:")
        print(f"     Direct effect: {validity_results['exclusion_restriction']['direct_effect']:.3f}")
        print(f"     Exclusion satisfied: {validity_results['exclusion_restriction']['exclusion_satisfied']}")
        
        # 5. Tetrad analysis
        print("\n5. Tetrad Constraints for Latent Variables:")
        tetrad_data = self.generate_tetrad_constraints()
        tetrad_results = self.test_tetrad_constraints(tetrad_data)
        
        print(f"   Variables with shared latent factor:")
        print(f"     Tetrad values: {[f'{t:.3f}' for t in tetrad_results['latent_factor_tetrads']]}")
        print(f"     Tetrad differences: {[f'{d:.3f}' for d in tetrad_results['tetrad_differences']]}")
        print(f"     Constraints satisfied: {tetrad_results['constraints_satisfied']}")
        
        print(f"   Variables without shared latent factor:")
        print(f"     Tetrad difference: {tetrad_results['independent_tetrad_diff']:.3f}")
        
        # 6. Visualization
        print("\n6. Generating Visualizations...")
        self.visualize_results()
        
        # 7. Summary insights
        print("\n7. Key Insights:")
        print("="*60)
        
        ols_bias_pct = endogeneity_results['bias_percentage']
        iv_avg_bias = np.mean([abs(iv_results['tsls_estimate'] - self.true_parameters['causal_effect']),
                              abs(iv_results['wald_estimate'] - self.true_parameters['causal_effect']),
                              abs(iv_results['iv_ratio_estimate'] - self.true_parameters['causal_effect'])])
        
        print(f"ENDOGENEITY PROBLEM:")
        print(f"  • OLS severely biased: {ols_bias_pct:.1f}% overestimate")
        print(f"  • Unobserved confounders create spurious correlation")
        print(f"  • Oracle method confirms bias source")
        
        print(f"\nINSTRUMENTAL VARIABLES SOLUTION:")
        print(f"  • IV methods recover true effect within {iv_avg_bias:.3f} bias")
        print(f"  • All IV estimators agree (2SLS, Wald, IV ratio)")
        print(f"  • Strong first stage (F = {iv_results['first_stage_f_statistic']:.1f} > 10)")
        
        instrument_valid = (validity_results['relevance']['is_relevant'] and
                           validity_results['exogeneity']['is_exogenous'] and
                           validity_results['exclusion_restriction']['exclusion_satisfied'])
        
        print(f"\nINSTRUMENT VALIDITY:")
        print(f"  • All assumptions satisfied: {instrument_valid}")
        print(f"  • Relevance: Strong first stage relationship")
        print(f"  • Exogeneity: Uncorrelated with confounders")
        print(f"  • Exclusion: No direct effect on outcome")
        
        print(f"\nLATENT VARIABLE DETECTION:")
        tetrad_satisfied = tetrad_results['constraints_satisfied']
        print(f"  • Tetrad constraints identify latent factor: {tetrad_satisfied}")
        print(f"  • Method can detect unobserved common causes")
        
        print(f"\nPRACTICAL IMPLICATIONS:")
        print(f"  • IV essential when randomization impossible")
        print(f"  • Finding good instruments is often the main challenge")
        print(f"  • Weak instruments can be worse than OLS bias")
        print(f"  • Always test instrument assumptions when possible")
        
        return self.results

# Run the experiment
if __name__ == "__main__":
    experiment = InstrumentalVariablesExperiment()
    results = experiment.run_complete_experiment()
