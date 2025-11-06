# Invariant Risk Minimization (IRM) 

---

## 1. Research Background and Problem Motivation

The core idea of traditional machine learning is **Empirical Risk Minimization (ERM)** — assuming training and test sets are independently and identically distributed (i.i.d.).

However, in reality, data often comes from multiple different environments (domains), such as different time periods, regions, devices, or sampling mechanisms. There exists **distribution shift**, causing models to perform well on training data but **fail on test environments**.

**Example:** When distinguishing "cow" vs "camel" in training, if the training set has "cows often with grass background" and "camels often with sandy background", the model may learn "background color" rather than "animal shape", and fail in new environments (e.g., on beaches).

This stems from the model overly relying on **spurious correlations** rather than **causal correlations**.

The proposal of IRM is to enable machine learning models to **learn truly invariant causal rules** across different environments, thereby achieving true **Out-of-Distribution (OOD) Generalization**.

---

## 2. Core Ideas and Basic Assumptions

### 2.1 Multi-environment Setting

Data comes from multiple environment sets:

$$\mathcal{E}_{tr} = \{e_1, e_2, \ldots, e_k\},$$

Each environment $e$ provides a sample set:

$$D^e = \{(x_i^e, y_i^e)\}_{i=1}^{n_e} \sim P(X^e, Y^e).$$

The goal is to find a predictor $f$ that has low risk across all environments (including unseen environments):

$$R_{OOD}(f) = \max_{e \in \mathcal{E}_{all}} R^e(f), \quad R^e(f) = \mathbb{E}_{(X, Y^e)}[\ell(f(X^e), Y^e)].$$

### 2.2 Conceptual Core: Learning Invariant Predictive Factors

IRM proposes:

> Learn a data representation $\Phi(X)$ such that under this representation, the optimal predictor (classifier) remains the same across different environments.

Formally, if there exists $w : \mathcal{H} \to \mathcal{Y}$ such that:

$$w \in \arg\min_{\tilde{w}} R^e(\tilde{w} \circ \Phi), \quad \forall e \in \mathcal{E}_{tr},$$

then $\Phi$ induces an **invariant predictor** $w \circ \Phi$.

In other words, IRM does not seek distribution matching (like domain adaptation), but **requires the same optimal predictor across all environments**.

This is the "causal mechanism's non-environmental variation" statistical portrayal.

---

## 3. Methodological Derivation and Optimization Form

### 3.1 Theoretical Optimization Problem (IRM)

$$\min_{\Phi, w} \sum_{e \in \mathcal{E}_{tr}} R^e(w \circ \Phi) \quad \text{s.t.} \quad w \in \arg\min_{\tilde{w}} R^e(\tilde{w} \circ \Phi), \, \forall e.$$

This problem is **bi-level optimization**, with the inner layer constraint requiring $w$ to be optimal in each environment.

However, directly solving the bi-level problem is extremely difficult, so the paper proposes a practical solvable approximate form.

---

### 3.2 Practical Version: IRMv1

Approximately converting the constraint to a penalty term:

$$\min_{\Phi} \sum_{e \in \mathcal{E}_{tr}} R^e(\Phi) + \lambda \cdot \|\nabla_{w|w=1.0} R^e(w \cdot \Phi)\|^2,$$

where:

- $\Phi(X)$ represents the learned feature representation;
- $w = 1.0$ is a fixed standard classifier;
- The penalty term enforces that in each environment, the gradient is small, i.e., $w$ is simultaneously optimal in each environment;
- $\lambda$ balances empirical risk and invariance.

Intuitively:
- **ERM** minimizes average risk;
- **IRM** further requires the optimal classifier to remain consistent across different environments.

---

### 3.3 Penalty Term Interpretation: Dlin

Under the squared loss,

$$D_{lin}(w, \Phi, e) = \|\mathbb{E}[\Phi(X^e)\Phi(X^e)^\top]w - \mathbb{E}[\Phi(X^e)Y^e]\|^2,$$

which is the **normal equations** residual.

Therefore $D_{lin} = 0$ is equivalent to $w$ being optimal in environment $e$.

Finally, IRMv1's optimization objective can be written as:

$$L_{IRMv1}(\Phi) = \sum_e R^e(\Phi) + \lambda D_{lin}(1.0, \Phi, e).$$

---

## 4. Theoretical Analysis and Causal Interpretation

### 4.1 Connection with Causal Models

Assume the observed data arises from a **structural causal model (SEM)** generated as:

$$X_i \leftarrow f_i(PA_i, N_i), \quad Y \leftarrow f_Y(PA_Y, N_Y),$$

where $PA_Y$ represents the causal parent set of $Y$.

When environments apply different interventions $e$ (changing some partial structural formulas), **if these interventions do not act on $Y$**, then:

$$P(Y \mid X_{PA_Y}) \text{ remains invariant across all environments.}$$

Therefore, if the learned stable predictor $w \circ \Phi$ responds to causal mechanisms:

**Conclusion:**
> When $\Phi$ captures $Y$'s causal parent information, IRM's learned predictor will naturally remain stable across all environments.

---

### 4.2 Main Theorem (Theorem 9 Overview)

Under linear conditions, assume:

$$Y^e = Z_1^\top \gamma + \varepsilon^e, \quad Z_1^\top \perp \varepsilon^e, \quad X^e = S(Z_1^e, Z_2^e),$$

where $Z_2$ is a non-causal variable.

If there exists an $r$-representation $\Phi$ and it satisfies the **linear general position** across training environments, then:

> If $\Phi$ induces an invariant predictor across training environments,
> then this predictor will remain invariant across all possible environments.

That is, IRM can generalize the learned invariance to all environments under finite environment scenarios, **forming true OOD generalization**.

---

### 4.3 Theoretical Insights

1. **IRM's learned predictor is based on $Y$'s causal parent set**;
2. **ERM's learned predictor may mix spurious features**;
3. **IRM is more robust**:
   - Robust Risk Minimization ⇒ plug-in insurance across training environments;
   - IRM ⇒ extrapolation to unknown environments;
4. **IRM can be viewed as "causal = one-time correct prediction"**.

---

## 5. Experimental Results and Empirical Observations

### 5.1 Synthetic Experiments

In linear latent variable models with significant lines (causal invariant $Z_1$ and non-causal feature $Z_2$),
IRM can accurately identify the causal direction, significantly better than ERM and ICP.

Particularly under "scrambled" observation conditions (where features are mixed with confounders), IRM can still recover the causal system similar to the oracle model.

### 5.2 Colored MNIST Experiment

On MNIST with artificially introduced color-label correlations (red/green color related):

- **ERM model** has high dependency on color;
- **IRM model** learns to rely on shape, only using form information;
- **Test accuracy improved from 17% for ERM to 67% for IRM**, approaching the oracle model that "removes color".

---

## 6. Philosophy and Methodological Significance of IRM

IRM's core lies in **transforming the causal invariance principle** into an **optimizable statistical learning objective**.

It bridges the gap between traditional ERM and causal learning:

- **From ERM to IRM**: from "matching training distributions" to "learning cross-environment stable rules";
- **From statistical causality**: from "relative to = mechanism stability";
- **From classical philosophy**: echoes "scientific laws should not change with observers" — **invariance can discover laws**.

---

## 7. Method Comparison and Relations

| Feature | ICP | Causal Transfer Learning | Anchor Regression | IRM |
|---------|-----|--------------------------|-------------------|-----|
| Goal | Identify causal parent set $P_{A_Y}$ | Learn transferable invariant model | Balance empirical fit and causal robustness | Learn predictor that does not vary across environments |
| Assumption | $P(Y \mid X_{S^\*})$ remains invariant across environments | $X_{S^\*}$ invariant across environments | Non-causal child variable $S^\*$ exists | Descriptive variable has 4 control disturbances |
| Mathematical Form | Invariance testing + hypothesis test | Linear regression + invariance residuals | Regularized risk + anchor penalty | Multi-environment risk + one-data descent constraint |
| Theoretical Foundation | Causal identifiability | Causal migration | Distributional robustness | Causal structure decomposes modular |
| Output | Causal parent variable confidence interval | Stable prediction model | Stable linear parameters/estimator | Invariant representation and predictor |
| Capability | Causal discovery | Migration generalization | Robust prediction | OOD extrapolation |
| Philosophy Direction | From causality → prediction | Causality supports prediction | Prediction → causality | From prediction → causal mechanism learning |

---

## 8. Main Conclusions and Insights

1. **IRM establishes a bridge from "learning representations" to "discovering causal rules"**: achieving invariant constraints through optimization to enable models to automatically aggregate causal features.

2. **Can extrapolate beyond training environments**: can generalize to unseen distributions for robust prediction.

3. **IRM's penalty term has clear causal interpretation**: forcing optimal mechanisms to remain consistent across environments.

4. **Compared with traditional Domain Adaptation, it's more fundamental**: not seeking distribution matching, but requiring the predictor itself to be invariant.

5. **Has strong interpretability and practicality in practical tools**: addressing biases, batch effects, and other OOD problems, achieving excellent results.

> **Core Insight:**
> True generalization does not come from learning "in different worlds but naturally established rules" —
> Machine learning should pursue universality, **not just learn from fitting data but discover invariance from change**.

---
