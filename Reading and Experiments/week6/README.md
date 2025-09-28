# Week 6: Causality in Machine Learning Tasks

## Overview

This experiment explores the application of causal inference principles to three core machine learning challenges: confounding bias removal, policy evaluation under selection bias, and domain adaptation. The implementation demonstrates both the power and limitations of causal methods in realistic scenarios.

## Learning Objectives

- Implement Half-Sibling Regression Model (HSRM) for confounding control
- Apply Inverse Propensity Weighting (IPW) in contextual bandit settings
- Design domain adaptation strategies based on causal invariance
- Evaluate causal methods against standard machine learning approaches

## Experimental Setup

### 1. Half-Sibling Regression Model (HSRM)
```
Confounders: C1, C2 (unobserved genetic/environmental factors)
Sibling pairs sharing confounders:
- X1 (causal) ↔ X2 (non-causal) share C1
- X3 (causal) ↔ X4 (non-causal) share C2
Independent features: X5 (causal), X6 (non-causal)

Y = 1.5*X1 + 0*X2 + 1.2*X3 + 0*X4 + 0.8*X5 + 0*X6 + confounding_effects
```

### 2. Causal Bandit Environment
```
Actions: {0, 1, 2} with true effects [2.0, 1.5, 1.0]
Selection bias: P(action|context, confounder)
Confounded rewards with unobserved confounder U
```

### 3. Transfer Learning Scenario
```
Source domain: X ~ N(0, 1)
Target domain: X ~ N(μ_shift, σ_scale)
Causal invariance: Same P(Y|X) across domains
Domain shifts: μ ∈ [-0.8, 1.2], σ ∈ [0.69, 1.60]
```

## Results Summary

### HSRM Performance
- **Standard regression bias**: 0.356
- **HSRM bias**: 0.351 
- **Oracle bias**: 0.009
- **Improvement**: 1.5%
- **Key insight**: Modest improvement due to moderate confounding strength

### Causal Bandit Results
- **Naive approach bias**: 0.230
- **Causal IPW bias**: 0.048
- **Improvement**: 78.9%
- **Key insight**: IPW effectively removes selection bias

### Transfer Learning Outcomes
- **Standard transfer MSE**: 0.140
- **Best adapted MSE**: 0.141
- **Improvement**: -0.39% (negative)
- **Key insight**: Standard methods handle moderate domain shift adequately

## Key Findings

1. **Context-dependent effectiveness**: Causal methods excel when their target bias is present (selection bias) but show limited gains when confounding is moderate

2. **Robust baseline performance**: Modern ML methods are surprisingly robust to moderate distribution shifts, limiting the gains from causal adaptation

3. **Selection bias most addressable**: IPW consistently delivers substantial improvements in bandit settings where selection mechanisms create clear bias

4. **HSRM requires strong confounding**: Half-sibling approaches need substantial shared confounding to demonstrate clear advantages over standard regression

## Practical Implications

- **Bandit applications**: Always use causal methods when selection bias is suspected
- **Observational studies**: HSRM beneficial when sibling structures exist and confounding is strong  
- **Domain adaptation**: Evaluate whether standard methods suffice before implementing complex causal approaches
- **Method selection**: Match causal approach to specific bias type present in data

## Files

- `fixed_causality_ml_tasks.py` - Complete experimental implementation
- `week6_causality_results.png` - Comprehensive visualization

## How to Run

```bash
python causality_ml_tasks.py
```

## Technical Notes

- Uses Ridge regression for numerical stability in transfer learning
- Implements correlation-based sibling identification for HSRM
- Applies weight clipping in IPW to prevent extreme influence
- Generates realistic domain shifts avoiding artificial performance gaps
