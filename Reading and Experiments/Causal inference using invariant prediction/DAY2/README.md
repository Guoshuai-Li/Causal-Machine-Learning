# ICP Experiment

## Overview

This experiment demonstrates the core principle of **Invariant Causal Prediction (ICP)** and explores its failure modes under model misspecification.

**Objectives:**
1. Verify ICP's ability to identify true causal parents from independent variables in linear Structural Equation Models (SEMs)
2. Explore ICP's failure mode when the linearity assumption is violated

---

## Causal Structure

### Experiment 1: Linear SEM
```
X1 → Y  (true causal relationship, coefficient = 1.5)
X2 ⊥ Y  (independent variable, no relationship with Y)
```

**Data Generation:**
- `X1 ~ N(0, 1)` (causal parent)
- `X2 ~ N(0, 1)` (independent variable)
- `Y = 1.5 × X1 + ε`, where `ε ~ N(0, 0.1²)`

**Environments:**
- Environment 1: Observational data
- Environment 2: Shift intervention on X1 (mean +2.0)
- Environment 3: Scale intervention on X1 (×3)


### Experiment 2: Nonlinear SEM
```
Y = 1.5 × X1 + 0.5 × X1² + ε  (quadratic relationship)
X2 ⊥ Y  (independent)
```

**Violation:** The true relationship is quadratic, but ICP assumes linearity.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `rpy2` | Python-R interface to call R's `InvariantCausalPrediction` package |
| `numpy` | Data generation and numerical operations |
| `matplotlib` | Visualization of residual patterns |
| `sklearn.linear_model` | Linear regression for residual analysis |


---

## Experimental Results

### Experiment 1: Linear SEM

**ICP Result:**
```
Accepted sets: [1]
```

**Residual Pattern:**
- Random scatter around zero → confirms linearity assumption holds
- Variance appears homogeneous across X1 values → supports invariance

### Experiment 2: Nonlinear SEM

**ICP Result:**
```
Accepted sets: []
```

**Why does ICP fail?**
1. True model: `Y = 1.5×X1 + 0.5×X1²`
2. ICP assumes: `Y = β×X1 + ε`
3. Linear regression residuals: `ε = 0.5×X1² + noise`
4. Residual distribution **varies** with X1 → violates invariance assumption

**Residual Pattern:**
- Clear **parabolic structure** (U-shape) visible in the plot
- Residuals are systematically negative near X1=0 and positive at extremes
- Pattern is **not random** → evidence of nonlinearity
- This structured residual pattern causes ICP to correctly reject the linear model

---


## Limitations of This Experiment

1. **Sample size:** 2000 samples per environment may be insufficient for very weak effects
2. **Environment diversity:** Only 3 environments; more environments could improve power
3. **Simplicity:** Real-world causal structures are often more complex
4. **Linearity assumption:** ICP requires correct model specification

---


## ICP Core Algorithm (Conceptual)

The `InvariantCausalPrediction` R package implements the following logic:

### 1. Candidate Set Generation
For p variables, enumerate all 2^p possible subsets S ⊆ {1, 2, ..., p}

### 2. For Each Candidate Set S:
```
a) For each environment e:
   - Fit linear regression: Y^(e) = X_S^(e) × β_S^(e) + ε^(e)
   - Compute residuals: R^(e) = Y^(e) - Ŷ^(e)

b) Test invariance hypothesis:
   H0: β_S^(1) = β_S^(2) = ... = β_S^(E)  (coefficients are equal)
   H0: Var(ε^(1)) = Var(ε^(2)) = ... = Var(ε^(E))  (residual variance is equal)

c) Use statistical tests (e.g., F-test, Levene's test) to test H0

d) If H0 is not rejected (p-value > α):
   → Accept S as a valid invariant predictive set
```

### 3. Output
Return all accepted sets S that pass the invariance test


---

## Source Code Reference

The R implementation can be found at:
- CRAN: https://cran.r-project.org/package=InvariantCausalPrediction
- Paper: Peters, J., Bühlmann, P., & Meinshausen, N. (2016). "Causal inference by using invariant prediction: identification and confidence intervals." *Journal of the Royal Statistical Society: Series B*, 78(5), 947-1012.

**Key functions:**
```r
ICP(X, Y, ExpInd, alpha = 0.05, ...)
  # X: predictor matrix (n × p)
  # Y: response vector (n × 1)  
  # ExpInd: environment index (n × 1)
  # alpha: significance level
  # Returns: accepted sets and confidence intervals
```

### Core Test Implementation (Simplified)

```r
# Pseudo-code of ICP's invariance test
for each candidate set S:
  for each environment e:
    beta[e] = lm(Y[e] ~ X[e, S])$coefficients
    residuals[e] = Y[e] - predict(lm(Y[e] ~ X[e, S]))
  
  # Test 1: Coefficient equality
  p_coef = test_coefficient_equality(beta[1], ..., beta[E])
  
  # Test 2: Residual variance homogeneity
  p_resid = levene.test(residuals[1], ..., residuals[E])$p.value
  
  # Accept if both tests pass
  if (p_coef > alpha & p_resid > alpha):
    accepted_sets.append(S)
```

**Critical insight:** The algorithm simultaneously tests both:
1. **Structural stability** (coefficients don't change)
2. **Distributional stability** (residual distribution doesn't change)

This dual testing makes ICP robust to various forms of confounding and model misspecification.

---


## Conclusion

This minimal experiment successfully demonstrates:

1. **ICP works when assumptions hold:** In the linear case, ICP correctly identifies X1 as a causal parent while excluding the independent variable X2 (by not accepting {2} alone)

2. **ICP fails safely when assumptions break:** In the nonlinear case, ICP rejects all candidates rather than producing false causal claims

3. **Residual diagnostics are essential:** Visual inspection of residuals provides immediate insight into whether the linearity assumption is satisfied

**Practical implication:** ICP is a powerful tool for causal discovery, but requires:
- Multiple diverse environments (interventional data)
- Correct model specification (linearity or appropriate transformations)
- Sufficient sample size for statistical power
- Careful residual diagnostics to validate assumptions

---
