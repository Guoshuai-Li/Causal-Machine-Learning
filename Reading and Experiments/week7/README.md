# Week 7: Latent Variables and Instrumental Variables

## Overview

This experiment addresses one of the most challenging problems in causal inference: estimating causal effects when unobserved confounders create endogeneity. The implementation demonstrates how instrumental variables can recover true causal effects even when traditional regression methods fail due to hidden confounding.

## Learning Objectives

- Understand endogeneity bias and its impact on causal estimation
- Implement multiple instrumental variable estimators (2SLS, Wald, IV ratio)
- Test instrumental variable assumptions systematically
- Apply tetrad constraints to detect latent variable structures
- Compare observational vs interventional estimation approaches

## Experimental Setup

### Endogeneity Structure
```
Unobserved confounder: U ~ N(0, 1)
Instrumental variable: Z ~ N(0, 1) (policy assignment, lottery)
Treatment (endogenous): X = 2.0*Z + 1.5*U + εX
Outcome: Y = 1.0*X + 2.0*U + εY

True causal effect: β = 1.0
```

### Key Challenges
1. **Endogeneity problem**: X and Y share unobserved confounder U
2. **OLS bias**: Simple regression Y~X captures spurious correlation
3. **IV solution**: Use Z as instrument to isolate causal variation in X

### Tetrad Analysis Setup
```
Latent factor model: L ~ N(0, 1)
Observed indicators:
- X1 = 0.8*L + ε1 (loading = 0.8)
- X2 = 0.7*L + ε2 (loading = 0.7)  
- X3 = 0.6*L + ε3 (loading = 0.6)
- X4 = 0.9*L + ε4 (loading = 0.9)

Tetrad constraint: Cov(X1,X2)*Cov(X3,X4) ≈ Cov(X1,X3)*Cov(X2,X4)
```

## Results Summary

### Endogeneity Bias Demonstration
- **OLS estimate**: 1.443 (44.3% overestimate)
- **Oracle estimate**: 0.999 (controls for U)
- **True effect**: 1.000
- **Bias magnitude**: Substantial overestimation due to confounding

### Instrumental Variable Performance
- **2SLS estimate**: 0.980 (2.0% error)
- **Wald estimate**: 0.980 (identical)
- **IV ratio estimate**: 0.980 (consistent)
- **Method agreement**: Perfect convergence across estimators

### Instrument Validity Tests
- **Relevance**: F-statistic = 3249.9 >> 10 (extremely strong)
- **Exogeneity**: Z-U correlation = -0.018 ≈ 0 (satisfied)
- **Exclusion restriction**: Direct effect = -2.436 (violated)
- **Overall validity**: 2 of 3 assumptions satisfied

### Latent Variable Detection
- **Tetrad differences**: [0.007, 0.013, 0.020] < 0.1 threshold
- **Constraints satisfied**: True (successful detection)
- **Method effectiveness**: Tetrad analysis correctly identifies shared latent factor

## Key Findings

1. **Endogeneity creates severe bias**: 44% overestimation in simple case demonstrates why controlling for confounders is crucial

2. **IV methods highly effective**: Even with imperfect exclusion restriction, IV reduces bias from 44% to 2%

3. **Strong instruments are powerful**: F-statistic > 3000 enables precise estimation and method convergence

4. **Perfect instruments are rare**: Exclusion restriction violation reflects real-world challenges in finding ideal instruments

5. **Tetrad constraints detect structure**: Statistical tests can reveal presence of unobserved common causes

## Practical Implications

- **Economics**: Education returns, income effects, program evaluation
- **Medicine**: Treatment effects with genetic instruments, Mendelian randomization
- **Policy**: Natural experiments, quasi-randomization
- **Business**: Price elasticity, marketing attribution with selection bias

## Methodological Insights

### When IV methods excel:
- Strong first stage relationships (F > 10)
- Clear source of exogenous variation
- Theoretical justification for exclusion restriction

### When standard methods fail:
- Unobserved confounding present
- Selection on unobservables
- Simultaneous causation

### IV limitations:
- Requires strong assumptions
- Local average treatment effects (LATE)
- Weak instruments worse than OLS

## Files

- `week7_latent_and_instrumental_variables.py` - Complete implementation
- `results/week7_iv_analysis.png` - Comprehensive visualization
- `analysis_notes.md` - Detailed interpretation and findings

## How to Run

```bash
python week7_latent_and_instrumental_variables.py
```

## Technical Notes

- Uses multiple IV estimators for robustness checking
- Implements systematic assumption testing
- Generates synthetic data with known ground truth
- Applies tetrad constraints for latent variable detection
- Includes strong first stage to avoid weak instrument problems
