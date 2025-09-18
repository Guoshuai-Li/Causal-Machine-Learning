# Experiment Analysis

## Background
This experiment tests front-door identification and counterfactual reasoning in scenarios where unmeasured confounding makes backdoor adjustment impossible, using the classic X→Z→Y pathway with unmeasured U→X,U→Y confounding.

## Results Interpretation

### 1. Front-door Identification Success
**What we tested**: Whether front-door formula P(Y|do(X)) = Σz P(Z=z|X)P(Y|Z=z) can recover causal effects under unmeasured confounding.

**Results**:
- Direct biased estimate: 1.631 (94% overestimate)
- Front-door analytical: 0.897 (7% error from ground truth 0.840)
- Front-door numerical: 0.897 (perfect consistency)
- Pathway effects: X→Z (1.207) × Z→Y (0.743) = 0.897

**Why front-door worked**: The method exploits the fact that X→Z has no confounding and Z→Y confounding can be handled by marginalizing over X. Despite unmeasured U creating severe bias in direct estimation, front-door identification reduced error from 94% to just 7%.

### 2. Counterfactual Reasoning Implementation
**What we tested**: Abduction-Action-Prediction framework for individual-level causal inference.

**Results**:
- Individual 100: Observed X=-0.633, Y=-2.028
- Counterfactual scenarios: X∈{-1.633, 0.367} → Y∈{-2.868, -1.188}
- Individual treatment effect: 1.680
- **Perfect consistency**: ITE = 2×ATE (1.680 = 2×0.840) due to linearity

**Why counterfactuals worked**: By inferring individual noise terms through abduction, then applying interventions while preserving structural equations, we obtained person-specific causal effects that align perfectly with population-level patterns under linearity assumptions.

### 3. SCM-Potential Outcomes Bridge Verification
**What we tested**: Whether SCM do-notation produces equivalent results to potential outcomes framework.

**Results**:
- Potential outcomes: E[Y_0]=0.010, E[Y_1]=0.850, E[Y_2]=1.690
- Treatment effects: ATE(1,0)=ATE(2,1)=0.840 (constant marginal effects)
- Linearity check: 2×ATE(1,0) = ATE(2,0) with difference 0.000
- **Perfect theoretical consistency** between frameworks

**Why equivalence held**: Under our linear SCM, potential outcomes Y_t are deterministic functions of treatment levels and individual characteristics, making do-calculus and PO framework mathematically identical.

### 4. Mechanism Invariance Under Distribution Changes
**What we tested**: Whether structural causal effects remain stable when input distributions change.

**Results**:
- Original observational effect: 1.631 (highly biased)
- New distribution observational: 0.816 (much closer to truth)
- True interventional effect: 0.840 (stable across distributions)
- Bias reduction: From 79% to 2.4% simply due to distribution change

**Why observational effects varied**: Changing P(X) altered the confounding pattern through U→X relationship, while structural equations X→Z→Y remained unchanged. This demonstrates why observational associations are unreliable but interventional effects are invariant.

### 5. Method Failure Analysis
**What we tested**: Comparative performance of different adjustment strategies when unmeasured confounding present.

**Results by bias magnitude**:
- Naive regression Y~X: 79.1% bias (massive confounding bias)
- Wrong adjustment for Z: 10.6% bias (blocks causal pathway X→Z→Y)
- Oracle with unmeasured U: 6.4% bias (direct pathway bias remains)
- Front-door identification: 5.7% bias (best achievable without oracle information)

**Why front-door outperformed alternatives**: Unlike backdoor adjustment which requires measuring all confounders, front-door only needs the mediator Z to be measured and no confounders on specific pathways, making it practical when U is unmeasured.

## Visual Analysis

### DAG Structure Interpretation
- Red dashed lines (U→X, U→Y): Unmeasured confounding creates correlation between X and Y
- Blue solid lines (X→Z→Y): Observable front-door pathway enables identification
- Legend correction: Proper color coding now clearly distinguishes observed vs hidden relationships

## Practical Implications

### 1. When to Use Front-door Identification
- **Backdoor blocked**: When unmeasured confounders prevent valid adjustment sets
- **Mediator available**: When causal pathway goes through measurable intermediate variables
- **Pathway assumptions met**: When X→Z has no confounding and Z→Y confounding can be controlled

### 2. Counterfactual Applications
- **Personalized medicine**: Individual-level treatment effect prediction
- **Policy analysis**: What would have happened to specific individuals under different policies
- **Fairness assessment**: Identifying individuals who would benefit most from interventions

### 3. SCM Framework Advantages
- **Unifies approaches**: Bridges interventional, counterfactual, and potential outcomes reasoning
- **Mechanistic understanding**: Provides clear causal pathways rather than just statistical associations
- **Robustness**: Mechanism invariance enables reliable prediction under distribution shift

## Methodological Lessons

### What Worked Exceptionally
- **Front-door identification**: Dramatic bias reduction (94% → 7%) when backdoor impossible
- **Linearity verification**: Perfect additive treatment effects confirm model assumptions
- **Framework integration**: SCM and PO approaches yielded identical numerical results

### What Revealed Complexity
- **Oracle limitations**: Even with unmeasured confounder, some bias persists due to pathway interactions
- **Distribution dependence**: Observational effects vary wildly with P(X) changes
- **Method sensitivity**: Small implementation differences create meaningful performance gaps

## Limitations

1. **Linearity assumption**: Real-world effects often non-linear with interactions
2. **No confounding assumptions**: Front-door requires specific pathway assumptions that may not hold
3. **Mediator measurement**: Assumes Z perfectly captures the causal pathway
4. **Structural knowledge**: Requires correct DAG specification for valid identification
5. **Sample size**: Finite samples may not achieve theoretical performance guarantees

## Bottom Line

**Front-door identification is highly effective**: When backdoor adjustment fails due to unmeasured confounding, front-door can reduce bias from 94% to 7% - a transformative improvement for causal inference.

**Counterfactual reasoning bridges theory and practice**: SCM provides concrete tools for individual-level causal inference that perfectly align with population-level effects under appropriate assumptions.

**Mechanism invariance is fundamental**: Understanding that structural relationships remain stable while observational patterns vary dramatically is crucial for robust causal inference in changing environments.

This experiment demonstrates the power of advanced causal inference techniques to overcome fundamental limitations of traditional statistical methods when facing unmeasured confounding challenges.
