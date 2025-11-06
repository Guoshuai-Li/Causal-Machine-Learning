# Invariant Models for Causal Transfer Learning 

---

## 1. Research Background and Problem

Traditional supervised learning assumes training and test samples follow the same distribution (i.i.d.), but in reality, **distribution shift** is common:
Models perform well on source domains but fail to generalize to target domains.
To address this issue, researchers have proposed **Transfer Learning** and **Domain Generalization (DG)**.

- **Domain Generalization**: Only source domain data available, no target domain samples.
- **Multi-task Learning (MTL)**: Target domain provides a small number of labeled samples.

Previous DG methods often assume **covariate shift**:
> Across all tasks, the conditional distribution $P(Y \mid X)$ remains invariant, with differences only from changes in input distribution.

However, this assumption is too restrictive. Inspired by causal reasoning, the authors propose a more relaxed assumption:
> There exists only a subset of features $S^\*$ such that $P(Y \mid X_{S^\*})$ remains invariant across tasks.

This subset is called the **invariant subset**.
From a causal perspective, if $X_{S^\*}$ is the causal parent set of $Y$, then its conditional distribution naturally remains invariant under different environments or interventions.

---

## 2. Core Ideas and Assumptions

The authors' theory is built on a linear regression framework and proposes three core assumptions:

1. **(A1) Invariant Conditional**
   
   There exists a subset $S^\*$ such that the distribution of $Y^k \mid X^k_{S^\*}$ is consistent across all training tasks.

2. **(A1') Invariance to Test Task**
   
   This invariance still holds for the test task.
   This is an extrapolation assumption that cannot be verified from training data.

3. **(A2) Linear Model**
   
   $$Y^k = \alpha^\top X^k_{S^\*} + \varepsilon^k, \quad \varepsilon^k \perp X^k_{S^\*}, \quad \varepsilon^k \sim \varepsilon$$
   
   That is, on the invariant subset, there exists a linear relationship between output and input.

These assumptions imply:

Even if the overall $P(Y \mid X)$ varies across tasks, as long as we find a feature subset $S^\*$ satisfying the invariance condition, the model can remain stable on unknown distributions.

---

## 3. Method Overview

### 3.1 Domain Generalization (No Target Domain Labels)

The goal is to minimize expected error under an unknown test domain:

$$\min_\beta \, \mathbb{E}_{P_T}[(Y^T - \beta^\top X^T)^2]$$

Since the test distribution is not observable, the authors propose training using the invariant subset:

$$\hat{\beta}_{CS}(S^\*) = \arg\min_{\beta} \sum_{k=1}^D (Y^k - \beta^\top X^k_{S^\*})^2$$

The theoretical result (Theorem 1) shows:
This estimator is **adversarially optimal** across all test distribution families satisfying (A1'), i.e.:

$$\hat{\beta}_{CS}(S^\*) = \arg\min_\beta \sup_{P_T \in \mathcal{P}} \mathbb{E}_{P_T}[(Y^T - \beta^\top X^T)^2]$$

Furthermore, Proposition 2 proves:
When differences between tasks increase (e.g., more severe interventions), this method has lower average error than models that directly pool all task data.

---

### 3.2 Multi-task Learning (With Limited Target Domain Data)

When the target domain provides some labeled or unlabeled samples, the authors transform the problem into **missing data estimation**.
The core idea is:
- In source tasks, some features (non-$S^\*$) are missing;
- In target tasks, some labels are missing.

Through the **EM algorithm**, jointly optimize:
**E-step** estimates conditional expectations of missing variables;
**M-step** updates the covariance matrix to maximize likelihood:

$$\ell(\Sigma) = -\frac{1}{2}\sum_i \log|\Sigma_i| - \frac{1}{2}\sum_i Z_{obs,i}^\top \Sigma_i^{-1}Z_{obs,i}$$

Finally, regression parameters are obtained from the estimated joint distribution.
This method can combine "invariant knowledge" from training tasks with "local samples" from test tasks to achieve better transfer.

---

### 3.3 Automatic Discovery of Invariant Subsets

In practice, $S^\*$ is unknown, so the authors propose search algorithms (Algorithm 1 & 2):
1. Perform linear regression for each feature subset $S$;
2. Calculate distribution differences of residuals across tasks;
3. Use **Levene test** or **HSIC independence test** to determine if invariance is satisfied;
4. Select the subset with minimum validation error from all subsets passing the test.

If feature dimensionality is high, Lasso pre-screening or greedy search can be used.
This test based on residual distribution consistency shares the same idea as ICP testing in causal discovery.

---

## 4. Causal Interpretation and Theoretical Connections

If data comes from the same structural equation model (SEM):

$$X_j = f_j(PA_j, N_j)$$

and different tasks correspond to interventions on some $X_j$ without intervening on $Y$, then $P(Y \mid X_{PA_Y})$ remains invariant across all tasks.

Therefore, $S^\*$ satisfying (A1)-(A1') is the causal parent set of $Y$.

This reveals:
- **Invariant Conditional** reflects the stability of causal mechanisms;
- **Robustness of Transfer Learning** essentially comes from learning true causal mechanisms.

Relationship with *Invariant Causal Prediction (ICP, Peters et al. 2016)*:

| Aspect | ICP | This Paper's Method |
|--------|-----|---------------------|
| Goal | Discover causal parent set | Find most generalizable stable subset |
| Basis | Pure causal inference | Predictive generalization and minimum error |
| Output | Causally interpretable set | Practical robust prediction model |
| Application | Causal discovery | Transfer learning / DG / MTL |

---

## 5. Main Conclusions and Insights

1. **Partial feature invariance** is a more relaxed and practical assumption than covariate shift.
2. In adversarial settings, predictors using only the invariant subset have theoretical optimality.
3. When task differences increase, invariant models have lower average error and greater robustness than pooled models.
4. Viewing MTL as a missing data problem and solving with EM naturally integrates invariant and specific information.
5. The paper bridges causal modeling and transfer learning, providing causal interpretation for generalization theory.

> **Core Insight:**
> The essence of generalization ability lies in learning causal mechanisms that are independent of environmental changes,
> rather than merely fitting statistical correlations in data distributions.

---
