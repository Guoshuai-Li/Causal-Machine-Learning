# Week 2: Causal Direction and Semi-Supervised Learning

## Overview
This experiment explores bivariate causal direction identification methods and investigates how causal knowledge impacts machine learning performance under various scenarios.

## Learning Objectives
- Implement ANM and LiNGAM algorithms for causal direction discovery
- Compare prediction performance in causal vs anti-causal directions
- Evaluate robustness of causal methods under distribution shifts
- Understand practical challenges in causal machine learning

## Experimental Setup

### Data Generation
```
Dataset 1: X1 → Y1
X1 ~ N(0,1)
Y1 = 2*X1 + 0.3*X1² + ε1

Dataset 2: Y2 → X2  
Y2 ~ N(0,1)
X2 = 1.5*Y2 + 0.2*Y2² + ε2
```

### Key Questions
1. **Causal Direction**: Can ANM and LiNGAM correctly identify X→Y vs Y→X?
2. **ML Performance**: Does predicting in causal direction improve accuracy?
3. **Robustness**: Are causal methods more robust to covariate shifts?

## Results Summary

### Causal Direction Identification
- **ANM method**: 1/2 correct identifications
- **LiNGAM method**: 2/2 correct identifications
- **Result**: LiNGAM showed superior performance for our data types

### Semi-Supervised Learning Performance
- **Dataset 1**: Anti-causal direction performed better (MSE: 0.1257 vs 0.4295)
- **Dataset 2**: Causal direction performed better (MSE: 0.2026 vs 0.3775)
- **Result**: Mixed findings challenge simple theoretical expectations

### Covariate Shift Robustness
- **Dataset 1**: Causal invariance 95.1% better than traditional method
- **Dataset 2**: Causal invariance 91.4% better than traditional method
- **Result**: Dramatic and consistent robustness advantage

## Key Findings
1. **Method sensitivity**: Causal direction identification depends heavily on data characteristics
2. **Prediction complexity**: Causal direction doesn't always guarantee better ML performance
3. **Robustness advantage**: Causal invariance provides substantial benefits under distribution shift
4. **Practical reality**: Real-world causal ML requires careful method selection and validation

## Files
- `causal_direction_and_semi_supervised_learning.py` - Main experiment code
- `experiment_results.png` - Visualization output

## How to Run
```bash
python causal_direction_and_semi_supervised_learning.py
```

**Requirements**: numpy, pandas, scikit-learn, matplotlib, scipy
