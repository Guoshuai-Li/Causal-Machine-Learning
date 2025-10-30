# Invariant Causal Prediction (ICP)

- **Problem:** Models often fail under distribution shifts or interventions; correlations learned from one environment may break in another.

- **Goal:** Identify variables whose relationship with the target *remains invariant* across multiple environments.

- **Key Assumption:** The conditional distribution $P(Y \mid X_{S^\*})$ *stays the same under all environments if* $S^\*$ contains the true causal parents of $Y$.

- **Approach:** Test each candidate set $S \subseteq \{X_1, \ldots, X_p\}$ for invariance of $P(Y \mid X_S)$ across environments, and take the intersection of accepted sets.

- **Outcome:** A statistically valid subset $\hat{S} \subseteq S^\*$ of direct causal predictors and confidence intervals for their effects.

- **Intuition:** Causal mechanisms are stable — exploiting this stability turns causality into a testable statistical property.

# Invariant Causal Prediction (ICP) — Method Derivation (Self-contained)

## Setup:
We observe data from multiple environments $e \in \mathcal{E}$, each providing samples $(X^e, Y^e)$.
Distributions may differ across environments, but causal mechanisms should remain stable.

## Core Assumption:
There exists a subset $S^* \subseteq \{1, ..., p\}$ such that

$$P(Y^e \mid X_{S^\*}^e) = P(Y^{e'} \mid X_{S^\*}^{e'}), \quad \forall e, e' \in \mathcal{E}.$$

Equivalently, in a linear model

$$Y^e = \mu + X^e \gamma^\* + \varepsilon^e, \quad \varepsilon^e \sim F_\varepsilon, \varepsilon^e \perp X_{S^\*}^e,$$

where the noise distribution $F_\varepsilon$ is identical across environments.

## Key Idea:
The correct causal predictors $S^\*$ make the conditional distribution of $Y$ invariant under environment changes.

Any subset missing a true parent breaks this invariance.

## Statistical Formulation:
For every candidate set $S \subseteq \{1, ..., p\}$, test the null hypothesis

$$H_{0,S}: P(Y^e \mid X_S^e) \text{ is identical across all } e \in \mathcal{E}.$$

In linear settings, this corresponds to testing whether regression coefficients and residual distributions are stable across environments.

## Algorithmic Procedure:
1. For each subset $S$, fit regressions $Y^e \sim X_S^e$ for all environments.
2. Test invariance of coefficients and residuals across environments.
3. Keep all subsets where $H_{0,S}$ is not rejected.
4. Define the final causal set as

$$\hat{S} = \bigcap_{S: H_{0,S} \text{ not rejected}} S.$$

## Guarantees:
If tests are valid at level $\alpha$, ICP ensures

$$P(\hat{S} \subseteq S^\*) \geq 1 - \alpha,$$

meaning it will not include false causal variables.

Confidence intervals for coefficients $\gamma_j$ can also be constructed.

## Intuitive Summary:
ICP transforms the *philosophical principle* of causal invariance into a *testable statistical procedure*:

by searching for variable sets that make the conditional distribution of $Y$ stable across environments,

it identifies direct causes with statistical guarantees.

# Invariant Causal Prediction (ICP) — Theoretical Positioning and Unique Assumptions

## Theoretical Foundation:

ICP is grounded in the *causal invariance principle* — the idea that causal mechanisms remain stable across environments or interventions that do not directly affect the target variable.

This concept is rooted in the principles of **autonomy** and **modularity** from Structural Causal Models (SCMs):

each causal mechanism behaves as an independent law of nature that stays unchanged when other mechanisms are perturbed.

## Conceptual Shift:

Unlike classical causal discovery methods that rely on *conditional independence* to recover the entire DAG,

ICP focuses on a **single target variable** $Y$ and identifies its **direct causal parents** by exploiting invariance of $P(Y \mid X_S)$ across multiple environments.

It converts a philosophical idea — *stability under change* — into a testable statistical property.

## Comparison with Other Approaches:

| Method Type | Core Assumption | Main Goal | ICP's Distinction |
|-------------|-----------------|-----------|-------------------|
| Constraint-based (e.g., PC, FCI) | Faithfulness + CI tests | Recover global DAG | ICP replaces faithfulness with invariance, focusing on one target |
| Score-based (e.g., GES) | Comparable model scores | Optimize DAG structure | ICP avoids full graph search; targets causal parents of $Y$ |
| Functional-based (e.g., LiNGAM, ANM) | Specific functional form (non-Gaussian / additive) | Identify causal directions | ICP is model-agnostic; works in both linear and nonlinear cases |
| Potential Outcomes / IV | Known interventions ↓ | Estimate causal effect | ICP does not require known intervention targets |

## Unique Assumptions and Strengths:

### 1. Invariance instead of Faithfulness:

Only assumes stability of $P(Y \mid X_{S^\*})$, not the presence of all independences in data.

### 2. No Need for Known Interventions:

Requires only an environment index $e$; interventions can be implicit or unknown.

### 3. Statistical Validity:

Provides confidence sets and intervals with explicit error control $(1 - \alpha)$.

### 4. Model-Agnostic Design:

Can be applied in linear, nonlinear, or semi-parametric models as long as invariance holds.

## Why It Matters:

- Operationalizes the invariance principle into a **statistical testing framework**.
- Introduces **confidence in causal discovery**, making causal claims quantitatively reliable.
- Serves as a theoretical backbone for **distribution shift generalization** and **causal domain adaptation**.
- Provides a robust inference framework used in genetics, economics, and other applied sciences.

## Doctoral-Level Summary:

**ICP reframes causal discovery from "learning a DAG" to "testing stability across environments."**

It adopts a weaker and more practical assumption — *invariance under interventions* — allowing statistically valid identification of causal predictors **without knowing the intervention targets**.
