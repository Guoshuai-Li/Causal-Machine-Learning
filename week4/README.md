# Week 3 Practice 2: Front-door Identification and Counterfactual Reasoning

## Overview
This experiment demonstrates advanced causal inference techniques when backdoor adjustment fails due to unmeasured confounding, focusing on front-door identification and counterfactual reasoning using SCM framework.

## Learning Objectives
- Implement front-door identification when backdoor adjustment is impossible
- Master counterfactual reasoning using Abduction-Action-Prediction framework
- Bridge SCM and Potential Outcomes (PO) frameworks using do-notation
- Test mechanism invariance under distribution changes

## Experimental Setup

### Front-door Scenario Structure
```
U → X, U → Y (unmeasured confounding)
X → Z → Y (front-door pathway)

Structural Equations:
X = 0.8*U + εx
Z = 1.2*X + εz  
Y = 0.7*Z + 0.9*U + εy
```

### Key Questions
1. **Front-door vs Backdoor**: Can front-door identification recover causal effects when backdoor fails?
2. **Counterfactual Reasoning**: How do we compute individual-level counterfactual outcomes?
3. **SCM-PO Bridge**: Are SCM interventions equivalent to potential outcomes framework?
4. **Mechanism Invariance**: Do structural relationships remain stable under distribution shifts?

## Results Summary

### Front-door Identification Performance
- **Direct biased estimate**: 1.631 (94% overestimate due to confounding)
- **Front-door identification**: 0.897 (only 7% error from ground truth 0.840)
- **Pathway decomposition**: X→Z effect (1.207) × Z→Y effect (0.743) ≈ 0.897
- **Method validation**: Analytical and numerical approaches identical (0.897)

### Counterfactual Analysis Results
- **Individual 100**: X=-0.633 → Y=-2.028 (observed)
- **Counterfactual 1**: X=-1.633 → Z=-1.772 → Y=-2.868
- **Counterfactual 2**: X=0.367 → Z=0.628 → Y=-1.188
- **Individual Treatment Effect**: 1.680 (consistent with 2×ATE due to linearity)

### SCM-PO Framework Bridge
- **Potential outcomes**: E[Y_0]=0.010, E[Y_1]=0.850, E[Y_2]=1.690
- **Treatment effects**: ATE(1,0)=0.840, ATE(2,0)=1.680, ATE(2,1)=0.840
- **Linearity verification**: 2×ATE(1,0) = ATE(2,0) exactly (difference: 0.000)

### Method Comparison (Bias Magnitude)
- **Naive regression**: 79.1% bias (worst)
- **Wrong adjustment for Z**: 10.6% bias (blocks causal pathway)
- **Oracle with unmeasured U**: 6.4% bias (pathway interference remains)
- **Front-door identification**: 5.7% bias (best achievable)

## Key Findings
1. **Front-door rescues identification**: Reduces bias from 94% to 7% when backdoor adjustment impossible
2. **Counterfactuals enable individual inference**: SCM provides tools for person-specific causal reasoning
3. **SCM-PO equivalence confirmed**: do-notation and potential outcomes yield identical results
4. **Mechanism invariance demonstrated**: Structural effects stable while observational associations vary dramatically

## Files
- `frontdoor_cf_po.py` - Main experiment code
- `experiment_results.png` - DAG visualization with corrected legend
- `analysis.md` - Detailed findings and interpretation

## How to Run
```bash
python frontdoor_cf_po.py
```

**Requirements**: numpy, pandas, scikit-learn, matplotlib, networkx, scipy
