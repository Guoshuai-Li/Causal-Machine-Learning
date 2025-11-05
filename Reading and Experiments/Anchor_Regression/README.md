# Anchor Regression 

---

## 1. Research Background and Problem Motivation

In real-world data from different sources, distributions often vary across batches or time points. As a result, **training and test sets may have non-i.i.d. distributions**.

This **heterogeneity** causes models to perform poorly in new environments when following traditional stable methods **(Huber regression, mixed-effects models)** or simply pooling all data, which **lacks theoretically sufficient stability guarantees**.

On the other hand, **Causal Models** possess **natural invariance to prediction shifts**:

If we learn the true causal parent set $P_{A_Y}$ through causal learning, then under interventions or environmental changes:

$$P(Y \mid X_{P_{A_Y}}) \text{ remains invariant.}$$

This insight forms the basis of causal mechanism-driven prediction invariance established (Peters et al., 2016).

However, **completely invariant causality may not hold strictly in practice**. In real data (with mild violations), there might be "approximately invariant" prediction accuracy.

Therefore, this paper proposes a new method that **achieves a balance between prediction accuracy and causal stability**—

**Anchor Regression**: Uses external variables (anchors) as anti-interference tools, making **corrections to OLS**. From the trade-off between **OLS (best fit)** and **IV (Two-Stage Least Squares)** (causal inference), it achieves realistic stable predictions.

---

## 2. Core Ideas and Assumptions

### 2.1 Model Setup

Suppose the training data is distributed across $\mathbb{E}_{train}$, where:

- $X \in \mathbb{R}^d$: predictor variables
- $Y \in \mathbb{R}$: response variable
- $A \in \mathbb{R}^q$: external variables (anchors), introducing data heterogeneity
- $H$: latent or unmeasured confounders

Assuming the data arises from a Structural Equation Model (SEM):

$$\begin{pmatrix} 
X \\ Y \\ H
\end{pmatrix}
=
B 
\begin{pmatrix} 
X \\ Y \\ H 
\end{pmatrix} 
+ \varepsilon + MA,
$$

where $A$ is independent of $\varepsilon$, and $M$ is the **shift matrix**, describing how anchors affect the system.

---

### 2.2 Anchor Regression Objective Function

Introducing the projection matrix $P_A$: projects any variable to the space orthogonal to the anchors.

For parameter $\gamma > 0$, anchor regression aims to minimize across the entire population:

$$\hat{b}_\gamma = \arg\min_b \mathbb{E}_{train}\left[((I - P_A)(Y - X^\top b))^2\right] + \gamma \mathbb{E}_{train}\left[(P_A(Y - X^\top b))^2\right].$$

Under finite sample settings, the corresponding estimator is:

$$\hat{b}_\gamma = \arg\min_b \|(I - \Pi_A)(Y - Xb)\|_2^2 + \gamma\|\Pi_A(Y - Xb)\|_2^2,$$

where $\Pi_A = A(A^\top A)^{-1}A^\top$.

---

### 2.3 Three Special Cases

| γ | Corresponding Model | Meaning |
|---|---------------------|---------|
| 0 | Partialling out A (removing anchor effects) | Adjusts for anchor bias, close to homogeneous samples |
| 1 | OLS | No heterogeneity protection |
| ∞ | IV (Two-Stage Least Squares) | Complete protection against interventions, zero causal parents |

Therefore, **Anchor Regression interpolates between γ as a continuum**:

$$b_0 = b_{P_A}, \quad b_1 = b_{OLS}, \quad b_\infty = b_{IV}.$$

**As γ increases**, the model transitions from strong heterogeneity adjustment; **as γ decreases**, it biases towards empirical fit.

---

## 3. Theoretical Foundation: Distributional Robustness

### 3.1 Minimax Form

Anchor Regression is equivalent to the following **distributional optimization**:

$$\min_b \max_{v \in \mathcal{C}_\gamma} \mathbb{E}_v[(Y - X^\top b)^2],$$

where the disturbance set is:

$$\mathcal{C}_\gamma = \{v : vv^\top \preceq \gamma M\mathbb{E}[AA^\top]M^\top\},$$

corresponding to **limited strength shifts** in the shift matrix.

**Theorem 1**: Anchor Regression's objective function uniformly minimizes the **worst-case Mean Squared Error (MSE)** under the above constraint. Therefore, $\hat{b}_\gamma$ is **minimax optimal** under the specified shift range.

Intuitively:
- **Small γ** ⇒ requires distribution fit to be more stringent (low bias tolerance);
- **Large γ** ⇒ requires unknown distribution to be more stable (high bias tolerance).

---

### 3.2 Relationship with Causal Parameters

When anchor $A$ satisfies the IV condition (only affects $X$ but not $Y$), then:

$$\lim_{\gamma \to \infty} \hat{b}_\gamma = b_{causal}.$$

That is, anchor regression's limit is the **causal effect estimator**.

But this approach is more relaxed: **even if anchors are not completely valid instruments** (directly used with $Y$ or $H$), stability prediction can still be achieved under certain conditions.

---

## 4. Anchor Stability and Replicability

Anchor Regression does not only provide stable predictions, but can also be used to **evaluate the stability of the model under distribution shifts**.

### 4.1 Definition

If for all $\gamma$ the following holds:

$$b_0 = b_\gamma = b_\infty, \quad \forall \gamma,$$

the model satisfies **Anchor Stability**.

### 4.2 Theoretical Results

- **Theorem 3 (Prediction and Estimation Stability)**
  
  If $b_0 = b_\infty$, then under all shifts caused by $M$:
  
  $$\mathbb{E}_{train}[(Y - X^\top b_0)^2] = \mathbb{E}_v[(Y - X^\top b_0)^2],$$
  
  meaning $b_0$ is uniformly optimal across all class shifts.
  
  ⇒ **Prediction error remains constant under interventions**.

- **Theorem 4 (Causal Interpretation)**
  
  Under the condition of no hidden confounding and satisfiability of DAG structure,
  
  If $b_0 = b_\infty$, then:
  
  $$b_0 = b_\infty = \mathbb{E}[Y \mid do(X = x)] - \mathbb{E}[Y],$$
  
  meaning this parameter **has causal effect meaning**.
  
  Therefore, **"anchor stability" means "distribution replicability" and "true causal interpretation"**.

---

## 5. Empirical Verification and Interpretation

### 1. Experimental Verification

In multiple simulations and real causal datasets, anchor regression shows **better replicability than OLS and IV** under moderate violations, with better performance.

### 2. Direct Interpretation

- Anchor regression = "**anti-unknown interference rule of thumb**";
- **γ controls "protection strength"**: moving from optimal empirical causality via the least-cost continuous path;
- Anchor stability can be viewed as a **"virtually testable causal = one-time condition"**.

### 3. Quantile View (Quantile View)

The paper further proves (Lemma 1):

$$Q(\alpha) = \mathbb{E}[((I - P_A)(Y - X^\top b))^2] + \gamma\mathbb{E}[(P_A(Y - X^\top b))^2],$$

where $\gamma$ and $\chi^2$ are distributed with a quantile relationship.

That is, anchor regression can be seen as **minimizing high quantile or worst-case MSE risk under different environments** — achieving equilibrium.

---

## 6. Methodological Summary and Theoretical Connections

| Feature | Invariant Causal Prediction (ICP) | Invariant Models for Causal Transfer Learning | Anchor Regression |
|---------|-----------------------------------|-----------------------------------------------|-------------------|
| Goal | Identify causal parents $P_{A_Y}$ | Learn transferable invariant prediction model | Balance prediction accuracy and distributional robustness |
| Core Assumption | $P(Y \mid X_{S^\*})$ remains invariant across environments | $X_{S^\*}$ remains invariant across environments | External variable $S^\*$ exists (anchors can cause $P(Y \mid X)$ to change) |
| Mathematical Form | Invariance testing + hypothesis testing | Minimax optimization under assignment invariance + invariance testing | OLS + $\gamma \times$ anchor penalty term |
| Theoretical Perspective | Causal identifiability | Causal → transfer learning | Distributional robustness ← causal = one-time |
| Output | Causal parent variable set | Stable prediction model | Stable parameters, replicable coefficients |
| Representative Idea | Graph → prediction | Causal support → transfer ↓ | Prediction → causal = one-time planning |

---

## 7. Main Conclusions and Insights

1. **Anchor Regression builds a connection between OLS and causal models**: achieving empirical fit and robustness balance through adjustment of γ.

2. **Under anchor-based prediction, range-bound distributional robustness is most optimal**. It is a bridge between causal inference and robustness optimization.

3. **Anchor Stability provides a testable causal criterion**: when OLS and IV are consistent, it implies the model has causal interpretation and environmental stability.

4. **Widely applicable in batch effects, time migration, domain shift issues**, and **enhances replicability** in practice.

5. **Core Insight**:

   > The key to replicable prediction is not completely "removing interventions",
   > but "balancing fit with stability",
   > finding the optimal solution in the tradeoff between statistics and causality.

---
