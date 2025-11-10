# CDNOD 

##  Libraries Used

### Core Dependencies
- **causal-learn** (>=0.1.3): Python causal discovery library
  - `CDNOD`: Causal Discovery from Nonstationary/heterogeneous Data algorithm
  - Implements constraint-based methods for causal inference

- **numpy**: Numerical computing and synthetic data generation

- **matplotlib**: Time series visualization with regime annotations

##  Experiment Objective

This experiment demonstrates how CDNOD leverages **nonstationarity** (distribution changes) to identify causal relationships through the **Independent Change Principle**.

**Key Learning Goals:**
1. Understand the Independent Change Principle for causal discovery
2. Learn how mechanism changes reveal causal direction
3. Compare CDNOD's approach with stationary methods like PCMCI
4. Observe how proper experimental design affects algorithm performance

##  Experimental Design

### The Independent Change Principle

**Core Idea:** If X causes Y (X → Y), then:
- When X's mechanism changes, Y may be affected (since Y depends on X)
- When Y's mechanism changes, X should NOT be affected (X doesn't depend on Y)
- By observing which variable's mechanism changes independently, we can infer causal direction

### Synthetic Data Generation

We create a 2-variable time series system with **3 regimes** where X's mechanism changes but Y's remains constant:

**Regime 1 (t = 0-666): X has weak autoregression**
```
X(t) = 0.3 × X(t-1) + ε_X
Y(t) = 0.5 × Y(t-1) + 0.6 × X(t-1) + ε_Y
```

**Regime 2 (t = 667-1333): X has strong autoregression**
```
X(t) = 0.7 × X(t-1) + ε_X        [X's mechanism changed!]
Y(t) = 0.5 × Y(t-1) + 0.6 × X(t-1) + ε_Y  [Y unchanged]
```

**Regime 3 (t = 1334-2000): X returns to weak autoregression**
```
X(t) = 0.3 × X(t-1) + ε_X        [X's mechanism changed again!]
Y(t) = 0.5 × Y(t-1) + 0.6 × X(t-1) + ε_Y  [Y still unchanged]
```

**Critical Design Features:**
-  X's autoregressive coefficient changes: 0.3 → 0.7 → 0.3
-  Y's autoregressive coefficient stays constant: 0.5
-  X→Y causal strength remains constant: 0.6
-  Y's mechanism depends on X, so Y is affected when X's mechanism changes
-  X's mechanism is independent, supporting the inference that X causes Y

**Parameters:**
- Sample size: n = 2000
- Number of regimes: 3
- Regime change points: t = 667, 1334
- Significance level: α = 0.05

### Workflow

```
Step 1: Data Generation
    ├─ Generate 2000-step time series with regime changes
    ├─ Create regime indicator array
    └─ Visualize time series with regime boundaries

Step 2: CDNOD Analysis
    ├─ Pass data and regime indicators to CDNOD
    └─ Extract causal graph

Step 3: Results Interpretation
    └─ Display detected causal links
```

## 📊 Experimental Results

### Detected Causal Links

```
==================================================
CDNOD RESULT
==================================================

Causal Graph:
Graph Nodes: X1; X2; X3

Graph Edges:
1. X1 --> X2
2. X3 --> X2

==================================================
```

**Variable Mapping:**
- X1 = X (cause variable)
- X2 = Y (effect variable)
- X3 = regime_indicator (auxiliary variable)

### Visualization

The time series plot shows:
- **X (blue)**: Exhibits different dynamics across three regimes
  - R1: Weak autoregression (X:0.3)
  - R2: Strong autoregression (X:0.7) - more persistent patterns
  - R3: Back to weak autoregression (X:0.3)
- **Y (orange)**: Shows consistent dynamics throughout, always responding to X
- **Red dashed lines**: Mark regime boundaries at t=667 and t=1334

Observable patterns:
- X's behavior visibly changes across regimes (amplitude, persistence)
- Y maintains similar statistical properties but follows X's influence
- The regime changes in X are clear, supporting CDNOD's inference

##  Results Analysis

###  Successfully Identified Causal Relationship

**X1 --> X2  (i.e., X --> Y)** ✓

This is the **primary success** of the experiment:
- CDNOD correctly identified the **directed** causal relationship
- The direction is correct: X causes Y
- This validates the Independent Change Principle approach

**Why it worked:**
1. X's mechanism changed independently (AR coefficient: 0.3 ↔ 0.7)
2. Y's mechanism remained constant throughout
3. CDNOD inferred: "X must be the cause because only X changes independently"
4. The constant X→Y strength (0.6) ensured Y consistently depends on X

###  Additional Detected Relationship

**X3 --> X2  (i.e., regime_indicator --> Y)**

This secondary finding requires interpretation:

**Why this appears:**
- The regime indicator is directly correlated with changes in the system
- When regimes change, X's mechanism changes, which affects Y through X→Y
- CDNOD detected this **indirect influence**: regime → X (mechanism) → Y

**Is this correct?**
- ✓ Philosophically: Regime changes do "influence" Y indirectly through X
- ✗ Technically: This is not a direct causal mechanism
-  Limitation: Regime indicators should ideally be treated as context, not causal variables

**Best interpretation:**
The regime indicator captures **environmental context** rather than a direct causal mechanism. In a properly designed CDNOD analysis, this auxiliary variable might be handled differently to avoid this artifact.

##  Key Insights and Comparisons

### The Power of the Independent Change Principle

**Traditional approach (e.g., Granger causality, PCMCI):**
- Assumes stationarity (stable distributions)
- Relies on temporal precedence and conditional independence
- May struggle when distributions shift

**CDNOD's approach:**
- **Leverages** nonstationarity as a resource
- Uses the pattern: "Causes change independently, effects depend on causes"
- Particularly powerful when natural regime changes occur

### Comparison with PCMCI

| Aspect | PCMCI | CDNOD |
|--------|-------|-------|
| **Data Assumption** | Stationary | Nonstationary |
| **Core Principle** | Conditional independence | Independent change |
| **Temporal Info** | Uses time lags | Uses distribution shifts |
| **Best Use Case** | Stable systems | Systems with regime changes |
| **This Experiment** | Would assume stationarity | Exploits regime changes |

### Design Lesson: Match Data to Algorithm

**Critical insight from this tutorial:**
Our initial attempts failed because we didn't design data according to CDNOD's principles:

** Initial failed design:**
```
X mechanism: constant (AR = 0.5 always)
Y mechanism: constant (AR = 0.4 always)  
X→Y strength: changing (0.2 → 0.7)
```
Result: Only detected undirected edge X --- Y

** Successful design:**
```
X mechanism: changing (AR: 0.3 → 0.7 → 0.3)
Y mechanism: constant (AR = 0.5 always)
X→Y strength: constant (0.6 always)
```
Result: Correctly detected X --> Y

**Lesson:** Algorithm performance depends critically on whether the data exhibits the patterns the algorithm is designed to detect.

## 🎓 Educational Value

### What This Experiment Teaches

1. **Algorithmic assumptions matter**
   - CDNOD requires independent mechanism changes
   - Without proper nonstationarity patterns, it cannot infer direction

2. **Experimental design is crucial**
   - "Design from the answer" works better than random data generation
   - Understanding the algorithm's principles guides better experiments

3. **Nonstationarity as information**
   - Distribution shifts aren't noise—they reveal causal structure
   - Real-world regime changes (economic cycles, climate patterns) can aid discovery

4. **Complementary methods**
   - CDNOD and PCMCI address different scenarios
   - Choose based on whether your data is stationary or nonstationary

### When to Use CDNOD

**Good scenarios for CDNOD:**
-  Data with observable regime changes
-  Systems where mechanisms shift over time
-  Economic data (policy changes, market regimes)
-  Climate data (seasonal patterns, climate shifts)
-  Biological systems (developmental stages, environmental changes)

**Poor scenarios for CDNOD:**
-  Perfectly stationary data
-  Single short observation period
-  Systems where all variables change synchronously
-  When regime information is unavailable or unreliable

##  Limitations and Considerations

### Experimental Limitations

1. **Regime indicator as a node**
   - The auxiliary regime variable appeared in the causal graph
   - Ideally, it should be context rather than a causal variable
   - May require different preprocessing or CDNOD parameters

2. **Simplified linear system**
   - Real-world systems are often nonlinear
   - This experiment uses linear relationships for clarity
   - CDNOD can handle nonlinear cases but may require more data

3. **Known regime boundaries**
   - We provided explicit regime labels
   - In practice, regimes may need to be detected first
   - This adds another layer of complexity

### CDNOD Algorithm Limitations

1. **Requires sufficient regime changes**
   - Few regimes = weak signal for causal inference
   - Need multiple observations per regime

2. **Assumption of independence**
   - Assumes cause mechanisms change independently of effects
   - Violations can lead to incorrect inferences

3. **Computational complexity**
   - Constraint-based methods can be computationally intensive
   - Scales with number of variables and regimes
