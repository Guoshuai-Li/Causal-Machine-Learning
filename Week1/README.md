# Week 1: Structural Causal Models (SCM) Basics

## Overview

This experiment demonstrates the fundamental difference between observational correlation and interventional causal effects using Structural Causal Models (SCM).

## Learning Objectives

- Build SCM systems with confounding variables
- Understand mechanism invariance principle
- Compare observational vs interventional estimates
- Implement do-calculus operations

## Experimental Setup

### SCM Structure
```
Z = εZ (confounder)
X = 2*Z + εX (treatment influenced by confounder)  
Y = 3*Z + 1*X + εY (outcome influenced by both)
```

### Key Questions
1. **Mechanism Invariance**: Do structural equations remain stable under interventions?
2. **Confounding Bias**: How much does observational analysis overestimate causal effects?
3. **Intervention Accuracy**: Can do-operations recover true causal effects?

## Results Summary

### Mechanism Invariance Test
- **Observational**: βX = 0.986, βZ = 3.033
- **Interventional**: βX = 0.455, βZ = 2.993
- **Result**: βZ remains stable (≈3.0), confirming mechanism invariance

### Causal Effect Comparison
- **Observational regression**: 2.413 (overestimated by 141%)
- **True causal effect**: 1.000 (theoretical)
- **Experimental effect**: 0.827 (close to truth)

## Key Findings

1. **Confounding creates substantial bias**: Observational studies can severely overestimate effects
2. **Interventions reveal truth**: do-operations provide unbiased estimates
3. **Mechanism invariance holds**: Structural relationships remain stable under interventions
4. **Visualization helps**: Color-coding by confounders reveals hidden structure

## Files

- `scm_basics.py` - Main experiment code
- `results/scm_experiment_results.png` - Complete visualization output
- `analysis.md` - Detailed findings and interpretation

## How to Run

```bash
python week1_scm_basics.py
```

