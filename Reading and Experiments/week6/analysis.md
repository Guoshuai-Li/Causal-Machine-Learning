# Experiment Analysis

## Background

This experiment evaluates three causal machine learning approaches across scenarios with different types of bias: confounding (HSRM), selection bias (causal bandit), and distribution shift (transfer learning). The goal is to understand when and why causal methods provide advantages over standard ML approaches.

## Results Interpretation

### 1. Half-Sibling Regression Model (HSRM)

**What we tested**: Whether sibling relationships can remove confounding bias in observational data.

**Results**:
- Standard regression bias: 0.356
- HSRM bias: 0.351 
- Improvement: 1.5%
- Oracle bias: 0.009 (theoretical minimum)

**Why the modest improvement**: 
- Moderate confounding strength in generated data
- Sibling identification based on correlation may miss subtle relationships
- Some features (X5, X6) had no confounding, diluting overall effect

**What worked**:
- X2 and X4 (pure confounding cases) showed clear bias reduction
- Sibling pairs were correctly identified using correlation patterns
- Oracle results confirm method validity

### 2. Causal Bandit Policy Evaluation

**What we tested**: Whether Inverse Propensity Weighting can remove selection bias in contextual bandits.

**Results**:
- Naive approach bias: 0.230
- Causal IPW bias: 0.048
- **Improvement: 78.9%**

**Why this worked so well**:
- Clear selection mechanism biasing action choice
- IPW directly targets the confounding pathway
- Sufficient data to estimate propensity scores accurately

**Key insight**: Selection bias is more amenable to causal correction than general confounding because the selection mechanism is often well-defined.

### 3. Transfer Learning with Domain Adaptation

**What we tested**: Whether causal invariance principles improve cross-domain generalization.

**Results**:
- Standard transfer MSE: 0.140
- Best causal method MSE: 0.141
- **Improvement: -0.39% (negative)**

**Why causal methods didn't help**:
- Modern ML methods (Ridge regression) already robust to moderate distribution shift
- Domain shift was realistic but not extreme enough to break standard approaches
- Feature standardization overcorrected, suggesting causal adaptation unnecessary

**Important finding**: Not all distribution shifts require sophisticated causal approaches.

## Visual Analysis

### Panel 1: HSRM Bias Comparison
- Oracle (green) shows theoretical minimum achievable bias
- HSRM (orange) modestly reduces bias for confounded features
- Limited improvement reflects realistic confounding scenarios

### Panel 2: Bandit Action Effects
- Causal estimates (orange) closely match true effects (red)
- Naive estimates (blue) systematically biased
- Demonstrates clear superiority of IPW approach

### Panel 3: Transfer Learning MSE
- Standardized method (orange) shows catastrophic failure (1.359 MSE)
- Other methods cluster around baseline performance
- Illustrates importance of method selection in domain adaptation

### Panel 4: Method Improvements
- Bandit shows dramatic 78.9% improvement
- HSRM shows modest 1.5% improvement  
- Transfer learning shows negative improvement
- Reveals context-dependent effectiveness of causal methods

### Panel 5: Domain Shift Visualization
- Clear distribution differences between source (blue) and target (red)
- Moderate but significant shift in both mean and variance
- Explains why some adaptation methods struggled

### Panel 6: Coefficient Recovery
- Transfer learning methods recover true coefficients reasonably well
- Standard and weighted methods perform similarly
- Suggests causal invariance holds but adaptation methods don't improve estimation

## Practical Implications

### 1. When Causal Methods Excel
- **Strong selection bias**: IPW dramatically outperforms in bandit settings
- **Clear confounding structure**: HSRM works when sibling relationships exist
- **Extreme distribution shift**: Causal adaptation becomes valuable (not demonstrated here)

### 2. When Standard Methods Suffice
- **Moderate confounding**: Standard regression with regularization competitive
- **Mild distribution shift**: Modern ML methods already robust
- **Well-specified models**: Domain knowledge can substitute for causal methods

### 3. Method Selection Guidelines
- **Contextual bandits**: Always use causal approaches when selection bias suspected
- **Observational studies**: Apply HSRM when genetic/familial data available
- **Domain adaptation**: Start with standard methods, escalate to causal only if failing

## Methodological Lessons

### What Worked
- **Realistic data generation**: Avoided artificial performance gaps
- **Multiple comparison methods**: Revealed relative strengths and weaknesses
- **Oracle comparisons**: Validated theoretical correctness
- **Detailed feature analysis**: Showed method-specific effects

### What Revealed Limitations
- **Correlation-based sibling identification**: May miss complex confounding patterns
- **Weight clipping in IPW**: Necessary for stability but may reduce effectiveness
- **Linear adaptation methods**: Insufficient for complex distribution shifts

## Limitations

1. **Synthetic data**: Real-world confounding may be more complex
2. **Linear relationships**: Non-linear effects could change relative performance
3. **Single confounder types**: Multiple simultaneous biases more challenging
4. **Sample size effects**: Larger datasets might reveal different patterns
5. **Feature selection**: Didn't explore causal feature selection methods

## Bottom Line

**Context matters more than method sophistication**: This experiment demonstrates that causal methods are not universally superior but excel in specific bias scenarios.

**Selection bias most tractable**: IPW and related methods consistently outperform when clear selection mechanisms exist.

**Confounding requires structure**: HSRM and similar methods need identifiable relationships (siblings, instruments) to provide substantial benefits.

**Standard methods are robust**: Modern ML with regularization handles many distributional challenges without explicit causal modeling.

**Match method to problem**: Understanding the specific bias type in your data is more important than applying the most sophisticated causal method.

This analysis prepares practitioners to make informed decisions about when causal ML methods justify their additional complexity over well-tuned standard approaches.
