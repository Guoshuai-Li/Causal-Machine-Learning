import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)

class CausalMLPractice:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def generate_causal_data(self, n=1000):
        """Generate causal data: X -> Y and Y -> X scenarios"""
        # Scenario 1: X -> Y (X causes Y)
        X1 = np.random.normal(0, 1, n)
        noise1 = np.random.normal(0, 0.5, n)
        Y1 = 2 * X1 + X1**2 * 0.3 + noise1
        
        # Scenario 2: Y -> X (Y causes X) 
        Y2 = np.random.normal(0, 1, n)
        noise2 = np.random.normal(0, 0.5, n)
        X2 = 1.5 * Y2 + Y2**2 * 0.2 + noise2
        
        return (X1, Y1), (X2, Y2)
    
    def anm_method(self, X, Y):
        """ANM method: Additive Noise Model Y = f(X) + ε"""
        # Fit Y = f(X) using ML model
        rf_xy = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_xy.fit(X.reshape(-1, 1), Y)
        pred_y = rf_xy.predict(X.reshape(-1, 1))
        residual_xy = Y - pred_y
        
        # Fit X = g(Y) using ML model  
        rf_yx = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_yx.fit(Y.reshape(-1, 1), X)
        pred_x = rf_yx.predict(Y.reshape(-1, 1))
        residual_yx = X - pred_x
        
        # Calculate correlation between residuals and cause variables
        corr_xy = abs(stats.pearsonr(X, residual_xy)[0])
        corr_yx = abs(stats.pearsonr(Y, residual_yx)[0])
        
        return corr_xy, corr_yx
    
    def lingam_method(self, X, Y):
        """Simplified LiNGAM method: Linear causal discovery based on non-Gaussianity"""
        # Standardize data
        X_std = self.scaler.fit_transform(X.reshape(-1, 1)).flatten()
        Y_std = self.scaler.fit_transform(Y.reshape(-1, 1)).flatten()
        
        # Calculate linear regression residuals for both directions
        # X -> Y direction
        lr_xy = LinearRegression()
        lr_xy.fit(X_std.reshape(-1, 1), Y_std)
        residual_xy = Y_std - lr_xy.predict(X_std.reshape(-1, 1))
        
        # Y -> X direction  
        lr_yx = LinearRegression()
        lr_yx.fit(Y_std.reshape(-1, 1), X_std)
        residual_yx = X_std - lr_yx.predict(Y_std.reshape(-1, 1))
        
        # Calculate non-Gaussianity of residuals using kurtosis
        kurtosis_xy = stats.kurtosis(residual_xy)
        kurtosis_yx = stats.kurtosis(residual_yx)
        
        return abs(kurtosis_xy), abs(kurtosis_yx)
    
    def ssl_experiment(self, X, Y, test_size=0.3):
        """Semi-supervised learning experiment: Compare 'cause->effect' vs 'effect->cause' ML performance"""
        n = len(X)
        n_labeled = int(n * (1 - test_size) * 0.5)  # 50% of training data as labeled
        
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(
            X.reshape(-1, 1), Y, test_size=test_size, random_state=42
        )
        
        # Create semi-supervised setting: only part of training data has labels
        indices = np.random.choice(len(X_train), n_labeled, replace=False)
        X_labeled = X_train[indices]
        Y_labeled = Y_train[indices]
        X_unlabeled = np.delete(X_train, indices, axis=0)
        
        # Method 1: X -> Y prediction
        rf_xy = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_xy.fit(X_labeled, Y_labeled)
        pred_y = rf_xy.predict(X_test)
        mse_xy = mean_squared_error(Y_test, pred_y)
        
        # Method 2: Y -> X prediction
        Y_train_reshaped, Y_test_reshaped = Y_train.reshape(-1, 1), Y_test.reshape(-1, 1)
        X_train_flat, X_test_flat = X_train.flatten(), X_test.flatten()
        
        Y_labeled_reshaped = Y_labeled.reshape(-1, 1)
        X_labeled_flat = X_labeled.flatten()
        
        rf_yx = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_yx.fit(Y_labeled_reshaped, X_labeled_flat)
        pred_x = rf_yx.predict(Y_test_reshaped)
        mse_yx = mean_squared_error(X_test_flat, pred_x)
        
        return mse_xy, mse_yx, n_labeled
    
    def reweighting_comparison(self, X, Y):
        """Compare traditional reweighting vs causal mechanism invariance"""
        n = len(X)
        
        # Create distribution shift: change X distribution
        # Original data
        X_orig = X.copy()
        Y_orig = Y.copy()
        
        # Shifted data (simulating domain adaptation scenario)
        shift_factor = 1.5
        X_shifted = X * shift_factor + np.random.normal(0, 0.3, n)
        Y_shifted = Y.copy()  # Assume Y distribution unchanged
        
        # Traditional method: train directly on shifted data
        rf_traditional = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_traditional.fit(X_shifted.reshape(-1, 1), Y_shifted)
        pred_traditional = rf_traditional.predict(X_orig.reshape(-1, 1))
        mse_traditional = mean_squared_error(Y_orig, pred_traditional)
        
        # Causal invariance method: train on original data
        rf_causal = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_causal.fit(X_orig.reshape(-1, 1), Y_orig)
        pred_causal = rf_causal.predict(X_orig.reshape(-1, 1))
        mse_causal = mean_squared_error(Y_orig, pred_causal)
        
        return mse_traditional, mse_causal
    
    def run_complete_experiment(self):
        """Run complete experimental pipeline"""
        
        # 1. Generate data
        print("1. Generating causal data")
        (X1, Y1), (X2, Y2) = self.generate_causal_data()
        print(f"   Dataset 1: {len(X1)} samples")
        print(f"   Dataset 2: {len(X2)} samples")
        
        # 2. ANM method testing
        print("\n2. ANM method for causal direction identification:")
        print("   Dataset 1 (Ground truth: X1 -> Y1):")
        corr_xy1, corr_yx1 = self.anm_method(X1, Y1)
        print(f"   X1->Y1 direction residual correlation: {corr_xy1:.4f}")
        print(f"   Y1->X1 direction residual correlation: {corr_yx1:.4f}")
        
        print("\n   Dataset 2 (Ground truth: Y2 -> X2):")
        corr_xy2, corr_yx2 = self.anm_method(X2, Y2)
        print(f"   X2->Y2 direction residual correlation: {corr_xy2:.4f}")
        print(f"   Y2->X2 direction residual correlation: {corr_yx2:.4f}")
        
        # 3. LiNGAM method testing
        print("\n3. LiNGAM method for causal direction identification:")
        print("   Dataset 1:")
        kurt_xy1, kurt_yx1 = self.lingam_method(X1, Y1)
        print(f"   X1->Y1 direction residual kurtosis: {kurt_xy1:.4f}")
        print(f"   Y1->X1 direction residual kurtosis: {kurt_yx1:.4f}")
        
        print("\n   Dataset 2:")
        kurt_xy2, kurt_yx2 = self.lingam_method(X2, Y2)
        print(f"   X2->Y2 direction residual kurtosis: {kurt_xy2:.4f}")
        print(f"   Y2->X2 direction residual kurtosis: {kurt_yx2:.4f}")
        
        # 4. Semi-supervised learning experiment
        print("\n4. Semi-supervised learning comparison:")
        print("   Dataset 1:")
        mse_xy1, mse_yx1, n_labeled1 = self.ssl_experiment(X1, Y1)
        print(f"   Using {n_labeled1} labeled samples")
        print(f"   X1->Y1 prediction MSE: {mse_xy1:.4f}")
        print(f"   Y1->X1 prediction MSE: {mse_yx1:.4f}")
        
        print("\n   Dataset 2:")
        mse_xy2, mse_yx2, n_labeled2 = self.ssl_experiment(X2, Y2)
        print(f"   Using {n_labeled2} labeled samples")
        print(f"   X2->Y2 prediction MSE: {mse_xy2:.4f}")
        print(f"   Y2->X2 prediction MSE: {mse_yx2:.4f}")
        
        # 5. Reweighting vs causal invariance comparison
        print("\n5. Covariate shift scenario construction:")
        print("   Dataset 1:")
        mse_trad1, mse_causal1 = self.reweighting_comparison(X1, Y1)
        print(f"   Traditional method MSE: {mse_trad1:.4f}")
        print(f"   Causal invariance method MSE: {mse_causal1:.4f}")
        
        print("\n   Dataset 2:")
        mse_trad2, mse_causal2 = self.reweighting_comparison(X2, Y2)
        print(f"   Traditional method MSE: {mse_trad2:.4f}")
        print(f"   Causal invariance method MSE: {mse_causal2:.4f}")
        
        # 6. Visualize results
        self.visualize_results(X1, Y1, X2, Y2)
        
        # Store results for further analysis
        self.results = {
            'anm_dataset1': (corr_xy1, corr_yx1),
            'anm_dataset2': (corr_xy2, corr_yx2),
            'lingam_dataset1': (kurt_xy1, kurt_yx1),
            'lingam_dataset2': (kurt_xy2, kurt_yx2),
            'ssl_dataset1': (mse_xy1, mse_yx1),
            'ssl_dataset2': (mse_xy2, mse_yx2),
            'robustness_dataset1': (mse_trad1, mse_causal1),
            'robustness_dataset2': (mse_trad2, mse_causal2)
        }
    
    def visualize_results(self, X1, Y1, X2, Y2):
        """Visualize experimental results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Dataset 1 scatter plot
        axes[0,0].scatter(X1, Y1, alpha=0.6, s=20)
        axes[0,0].set_title('Dataset 1: X1 -> Y1')
        axes[0,0].set_xlabel('X1')
        axes[0,0].set_ylabel('Y1')
        
        # Dataset 2 scatter plot
        axes[0,1].scatter(X2, Y2, alpha=0.6, s=20)
        axes[0,1].set_title('Dataset 2: Y2 -> X2')
        axes[0,1].set_xlabel('X2')
        axes[0,1].set_ylabel('Y2')
        
        # Method comparison bar chart
        methods = ['ANM\n(X1->Y1)', 'ANM\n(Y1->X1)', 'LiNGAM\n(X1->Y1)', 'LiNGAM\n(Y1->X1)']
        corr_xy1, corr_yx1 = self.anm_method(X1, Y1)
        kurt_xy1, kurt_yx1 = self.lingam_method(X1, Y1)
        scores1 = [corr_xy1, corr_yx1, kurt_xy1, kurt_yx1]
        
        axes[1,0].bar(methods, scores1, color=['blue', 'red', 'green', 'orange'])
        axes[1,0].set_title('Dataset 1 Method Comparison')
        axes[1,0].set_ylabel('Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # SSL performance comparison
        mse_xy1, mse_yx1, _ = self.ssl_experiment(X1, Y1)
        mse_xy2, mse_yx2, _ = self.ssl_experiment(X2, Y2)
        
        ssl_methods = ['X1->Y1', 'Y1->X1', 'X2->Y2', 'Y2->X2']
        ssl_scores = [mse_xy1, mse_yx1, mse_xy2, mse_yx2]
        
        axes[1,1].bar(ssl_methods, ssl_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[1,1].set_title('Semi-supervised Learning Performance Comparison')
        axes[1,1].set_ylabel('MSE')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    experiment = CausalMLPractice()
    experiment.run_complete_experiment()
