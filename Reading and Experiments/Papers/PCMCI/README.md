# PCMCI 
---

## 1. Research Background and Motivation

In fields such as climate science and neuroscience, real-world forecasting often faces the challenge of observing large-scale synchronous column data to identify causal structures.

However, real-world data typically has the following characteristics:

- **High dimensionality** (hundreds of variables)
- **Self-correlation, non-linear relationships, heterogeneity**
- **Presence of non-linear relationships**
- **Limited sample size**

Traditional **Granger causality** testing is based on bi-directional VAR models and can improve dependency measures, but it is easily misled under high dimensionality:

- **Curse of dimensionality**
- **False positive rate** increases
- Difficulty handling non-linear relationships

Runge et al. proposed the **PCMCI method** (Peter-Clark Momentary Conditional Independence) to precisely address this challenge.

When maintaining causal interpretability and assumption control simultaneously, it is applicable to discovering synchronous causal networks in large-scale non-linear time series.

---

## 2. Core Problem and Theoretical Foundation

### 2.1 Problem Definition

**Objective**:  
In multivariate time series {X₁ᵗ, X₂ᵗ,..., Xₙᵗ},  
identify all directed causal relationships Xⁱₜ₋τ → Xʲₜ.

**Definition**:

Xⁱₜ₋τ ⊥̸ Xʲₜ | X₋ᵢ \ {Xⁱₜ₋τ}

If the above conditions are independent and are violated, then there is a causal relationship.

This framework is based on **Time Series Graphs** and **Causal Markov** conditions.

### 2.2 Limitations of Granger Causality

Granger causality's Full Conditional Independence (FullCI) form:

Xⁱₜ₋τ ⊥̸ Xʲₜ | X₋ₜ

Under high-dimensional conditions, controlling all other variables in the past will lead to:

- Insufficient degrees of freedom in conditional sets
- Reduced statistical power
- Frequent occurrence of **spurious associations**

PCMCI uses structured condition selection and double-stage independent testing mechanisms to significantly improve this problem.

---

## 3. Method Design: PCMCI Two-Stage Algorithm

PCMCI = **PC1 Condition Selection** (Condition Selection) + **MCI Condition Independence Test** (Momentary Conditional Independence Test)

### Stage 1: PC1 Condition Selection

**Objective**: Remove variables with no causal relationship to the target variable, reducing dimensionality.

**Algorithm Idea**:

**1. Initialize**: P̂(Xʲₜ) = X₋ₜ (all variables in the past).

**2. Iterative deletion**: If

Xⁱₜ₋τ ⊥ Xʲₜ | S

For a certain conditional set S holds, then remove this variable.

**3. Control significance level α** and only retain possible parent node sets.

This step is based on the **PC algorithm** (Spirtes & Glymour, 1991) discovery stage, optimized for applicable time series-related **PC1 variant**.

---

### Stage 2: MCI Conditional Independence Test

Under low-dimensional conditions, perform final verification for candidate results:

MCI: Xⁱₜ₋τ ⊥ Xʲₜ | P̂(Xʲₜ) \ {Xⁱₜ₋τ}, P̂(Xⁱₜ₋τ)

- Simultaneously control the parent nodes of the target variable and source variable's parent nodes
- Possess double elimination of mutual influence
- Provides statistically interpretable "**causal strength**"

### Selectable Independence Test Methods

PCMCI can flexibly configure multiple conditional independence tests:

| Test Type | Name | Characteristics |
|-----------|------|-----------------|
| Linear | ParCorr (partial correlation) | Efficient, suitable for linear relationships |
| Non-linear | GPDC (Gaussian Process Distance Correlation) | Suitable for non-linear and noise model |
| Non-parametric | CMI (Conditional Mutual Information) | Distribution-free, most suitable for complex systems |

---

## 4. Theoretical Properties

### (1) Consistency

Under the assumptions of Causal Markov and Causal Sufficiency,  
PCMCI can asymptotically recover the true causal network structure under unlimited sample conditions.

### (2) False Positive Control

MCI uses simultaneous control of self-correlation and conditional set structure, maintaining a significance level of 5% in true self-correlation variables while correctly controlling errors.

### (3) Effect Size Improvement

Compared to FullCI, PCMCI only truly relevant quantities are retained, avoiding "excessive elimination" (explaining away),  
improving statistical power for effect size of causal results.

### (4) Causal Strength Interpretability

MCI statistics (such as partial correlation value) can be visually ranked below the causal impact bandwidth threshold unit, which can be used for importance ranking.

---

## 5. Experimental Verification and Results

### 5.1 Climate Case Study

- **Data**: ENSO (Nino3.4 index) and North American temperature.
- **Results**: PCMCI detected the causal direction Nino_{t-2} → BCT_t under high-dimensional scenarios (adding irrelevant variables Z, W), with a detection rate > 80%.
- Significantly superior to FullCI (~ 40%).

### 5.2 Synthetic Data Experiments

Designed high-dimensional linear and non-linear networks, comparing the following method properties:

| Method | Dimensional Adaptability | Assumption Control | Detection Rate (TPR) | Non-linear Adaptability | Computational Efficiency |
|--------|--------------------------|-------------------|---------------------|------------------------|-------------------------|
| Correlation | ✗ | Poor | High false positives | ✗ | High |
| FullCI | ✗ | Good | As degrees of freedom decrease | Partial | Low |
| Lasso | ✓ | Possible | Medium | ✗ | Medium |
| PC | ✓ | Unstable (self-correlation) | Medium | Partial | Low |
| **PCMCI** | ✓✓✓ | **Stable** | **High (≥0.8)** | ✓✓✓ | **Medium-High** |

**Conclusion**:  
PCMCI maintains high measurement test accuracy and control limits when sample size is limited and dimensionality is high (Nτ > T).  
Its detection capability for non-linear relationships is significantly superior to traditional Granger methods.

---

## 6. Relationship with Granger Causality

| Comparison Dimension | Granger Causality (1969) | PCMCI (Runge et al., 2019) | Relationship and Evolution Direction |
|---------------------|-------------------------|---------------------------|-------------------------------------|
| **Core Idea** | Causality = prediction improvement: if Y's past can improve X's prediction, then Y → X | Causality = conditional independence failure: if Xⁱₜ₋τ and Xʲₜ are not independent under conditional set control, then Xⁱₜ₋τ → Xʲₜ | PCMCI will define Granger's prediction criterion as graph model definition based on conditional independence |
| **Mathematical Framework** | Linear autoregressive model (VAR) for multivariate comparisons | Condition independence testing under time series causal graph model (Time Series Graphs) | From parametric/multivariate tests to non-parametric statistical independence testing |
| **Applicable Data Type** | Low-dimensional, linear, stationary time series | High-dimensional, non-linear, self-correlated time series | PCMCI improves the application scope of Granger causality |
| **Dimensional Scalability** | Limited by double-counting with large number of variables, difficult precision area and regression | PC1 removes irrelevant paths through condition selection, significantly eliminating source and noise interference | From "complete prediction transformation" to "elimination of conditional causes" |
| **Statistical Tools** | F test / Wald test (comparison of quantity comparison estimation) | Conditional independence test (partial correlation, GPDC, CMI, etc.) | Expanded to non-linear testing and non-high-distribution |
| **Assumption Degree** | Stationarity, linear, unbiased reduction | Causal completeness, Causal Markov, Faithfulness | PCMCI moderately formalizes Granger's implicit assumptions in graph model framework |
| **Result Output** | Unidirectional determination (whether Granger causes X) | Complete time series causal network structure (multi-source, multi-time delay) | From binary verification → complete causal network reconstruction |
| **Main Advantages** | Simple, intuitive, interpretable | Scalable, highly interpretable, assumption control | PCMCI is a structured high-level implementation result of Granger thinking |
| **Main Limitations** | Unable to process non-linear and high-dimensional dependencies | For high/low-dimensional and non-stationary situations, still relies on completion assumption | PCMCI is problematic because of dependence on system completion assumptions |
| **Representative Implementation** | statsmodels.tsa.stattools.grangercausalitytests() (Python), lmtest::grangertest() (R) | tigramite (Python) library PCMCI and PCMCI+ implementation | Both are in mainstream time series causal research and experimental verification |

---

## 7. Method Contributions and Limitations

###  Main Contributions

1. Provides a unified framework linking graph models and time series causal inference
2. Achieves actual control of assumption control and rate output under non-linear and high-dimensional conditions
3. Provides interpretable causal strength measures
4. Open source implementation (Python package **tigramite**) with great practical utility

### Limitations

- Relies on **Causal Sufficiency** assumption, sensitive to unobserved confounders
- Only applicable to causal directions with time intervals
- Not suitable for short time series or verification based on very extreme connections
- Non-parametric testing (CMI) is computationally intensive

---

## 8. Conclusion and Insights

PCMCI provides a **scalable, interpretable, organizable** solution for causal discovery in dynamic systems.

It inherits the "prediction improvement" concept of Granger causality and evolves combined with graph model conditional independence frameworks, realizing a true transition:

**From correlation network → causal network**.

In contemporary time series causal learning systems, PCMCI is being considered as:

**"A standard baseline for high-dimensional non-linear time series causal discovery" (baseline)**.
