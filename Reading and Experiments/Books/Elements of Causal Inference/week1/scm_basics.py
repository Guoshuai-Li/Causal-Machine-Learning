# SCM Basics: Observational Correlation vs Interventional Effects
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Settings
np.random.seed(42)
plt.rcParams['font.family'] = 'Arial'

class SCMSimulator:
    def __init__(self, n=1000):
        self.n = n
        
    def confounded_system(self):
        """Confounded system: Z → X, Z → Y"""
        Z = np.random.normal(0, 1, self.n)
        X = 2 * Z + np.random.normal(0, 0.5, self.n)
        Y = 3 * Z + 1 * X + np.random.normal(0, 0.5, self.n)
        return pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})
    
    def direct_system(self):
        """Direct causal system: X → Y"""
        X = np.random.normal(0, 1, self.n)
        Y = 2 * X + np.random.normal(0, 0.5, self.n)
        return pd.DataFrame({'X': X, 'Y': Y})
    
    def intervention_doX(self, x_val):
        """Execute intervention do(X = x_val)"""
        Z = np.random.normal(0, 1, self.n)
        X = np.full(self.n, x_val)  # X fixed to constant
        Y = 3 * Z + 1 * X + np.random.normal(0, 0.5, self.n)
        return pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})

def spring_system_scm():
    """Spring system: F → X → Y (Hooke's Law)"""
    n = 800
    k = 0.5
    
    F = np.random.uniform(0, 10, n)
    X = k * F + np.random.normal(0, 0.2, n)
    Y = X + np.random.normal(0, 0.1, n)
    
    return pd.DataFrame({'Force': F, 'Compression': X, 'Observed': Y})

def verify_invariance_correct():
    """
    Mechanism invariance test: In observational vs do(X=const) environments,
    estimate β_Z, β_X using correct specification Y ~ X + Z, 
    both should be consistent (≈3 and ≈1).
    """
    n = 5000
    scm = SCMSimulator(n=n)
    # Environment A: Observational
    A = scm.confounded_system()   # Y = 3Z + 1*X + ε, and X = 2Z + ν
    # Environment B: do(X = 1.0)
    B = scm.intervention_doX(1.0) # Still Y = 3Z + 1*X + ε, but X no longer depends on Z
    
    def coef_Y_on_XZ(df):
        XZ = np.column_stack([np.ones(len(df)), df["X"].to_numpy(), df["Z"].to_numpy()])
        y  = df["Y"].to_numpy()
        beta, *_ = np.linalg.lstsq(XZ, y, rcond=None)
        # beta = [intercept, beta_X, beta_Z]
        return beta[1], beta[2]
    
    betaX_A, betaZ_A = coef_Y_on_XZ(A)
    betaX_B, betaZ_B = coef_Y_on_XZ(B)
    
    return (betaX_A, betaZ_A), (betaX_B, betaZ_B)

def observational_vs_interventional():
    """Core experiment: Compare observational correlation vs interventional effects"""
    scm = SCMSimulator()
    
    # Observational data
    obs_data = scm.confounded_system()
    obs_coef = stats.linregress(obs_data['X'], obs_data['Y']).slope
    
    # Intervention data: Calculate marginal causal effect
    int_data1 = scm.intervention_doX(0.0)
    int_data2 = scm.intervention_doX(1.0)
    
    # True causal effect: Marginal effect of X increasing by 1 unit on Y
    causal_effect = int_data2['Y'].mean() - int_data1['Y'].mean()
    
    return obs_data, obs_coef, causal_effect, int_data1, int_data2

def create_plots(obs_data, int_data1, int_data2, obs_coef, causal_effect):
    """Generate comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Observational scatter plot
    axes[0,0].scatter(obs_data['X'], obs_data['Y'], alpha=0.6, s=15)
    x_range = np.linspace(obs_data['X'].min(), obs_data['X'].max(), 100)
    axes[0,0].plot(x_range, obs_coef * x_range + obs_data['Y'].mean() - obs_coef * obs_data['X'].mean(), 'r--')
    axes[0,0].set_title(f'Observational Data\nRegression Coef: {obs_coef:.3f}')
    axes[0,0].set_xlabel('X')
    axes[0,0].set_ylabel('Y')
    
    # Colored by confounder
    scatter = axes[0,1].scatter(obs_data['X'], obs_data['Y'], c=obs_data['Z'], cmap='viridis', s=15)
    axes[0,1].set_title('Colored by Confounder Z')
    axes[0,1].set_xlabel('X')
    axes[0,1].set_ylabel('Y')
    plt.colorbar(scatter, ax=axes[0,1])
    
    # Intervention effects
    means = [int_data1['Y'].mean(), int_data2['Y'].mean()]
    axes[1,0].bar(['do(X=0)', 'do(X=1)'], means, alpha=0.7)
    axes[1,0].set_title(f'Marginal Causal Effect: {causal_effect:.3f}')
    axes[1,0].set_ylabel('E[Y|do(X)]')
    
    # Coefficient comparison
    categories = ['Obs Regression', 'Theoretical', 'Experimental']
    values = [obs_coef, 1.0, causal_effect]
    axes[1,1].bar(categories, values, color=['red', 'green', 'blue'], alpha=0.7)
    axes[1,1].set_title('Coefficient Comparison')
    axes[1,1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def main():
    # 1. Basic SCM systems
    scm = SCMSimulator()
    confounded_data = scm.confounded_system()
    direct_data = scm.direct_system()
    
    print("Confounded system data shape:", confounded_data.shape)
    print("Direct system data shape:", direct_data.shape)
    
    # 2. Physical system abstraction
    spring_data = spring_system_scm()
    print("Spring system data shape:", spring_data.shape)
    
    # 3. Verify mechanism invariance
    (betaX_obs, betaZ_obs), (betaX_int, betaZ_int) = verify_invariance_correct()
    print(f"Observational environment coefficients - βX: {betaX_obs:.3f}, βZ: {betaZ_obs:.3f}")
    print(f"Interventional environment coefficients - βX: {betaX_int:.3f}, βZ: {betaZ_int:.3f}")
    print(f"Mechanism invariance verification: βX difference={abs(betaX_obs-betaX_int):.4f}, βZ difference={abs(betaZ_obs-betaZ_int):.4f}")
    
    # 4. Core comparison experiment
    obs_data, obs_coef, causal_effect, int_data1, int_data2 = observational_vs_interventional()
    
    print(f"Observational regression coefficient (X→Y): {obs_coef:.3f}")
    print(f"Marginal causal effect (X increases by 1 unit): {causal_effect:.3f}")
    print(f"Theoretical marginal causal effect: 1.000")
    print(f"Reason for difference: Observational regression contains confounding bias")
    
    # 5. Visualization
    create_plots(obs_data, int_data1, int_data2, obs_coef, causal_effect)
    
    # Data summary
    print("\nExperiment Results Summary:")
    print(f"E[Y|do(X=0)]: {int_data1['Y'].mean():.3f}")
    print(f"E[Y|do(X=1)]: {int_data2['Y'].mean():.3f}")
    print(f"Observational correlation coefficient: {obs_data['X'].corr(obs_data['Y']):.3f}")

if __name__ == "__main__":
    main()
