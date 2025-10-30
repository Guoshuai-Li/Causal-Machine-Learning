# Invariant Causal Prediction (ICP)

- **Problem:** Models often fail under distribution shifts or interventions; correlations learned from one environment may break in another.

- **Goal:** Identify variables whose relationship with the target *remains invariant* across multiple environments.

- **Key Assumption:** The conditional distribution $P(Y \mid X_{S^\*})$ *stays the same under all environments if* $S^\*$ contains the true causal parents of $Y$.

- **Approach:** Test each candidate set $S \subseteq \{X_1, \ldots, X_p\}$ for invariance of $P(Y \mid X_S)$ across environments, and take the intersection of accepted sets.

- **Outcome:** A statistically valid subset $\hat{S} \subseteq S^\*$ of direct causal predictors and confidence intervals for their effects.

- **Intuition:** Causal mechanisms are stable — exploiting this stability turns causality into a testable statistical property.

# Method Derivation
- **Setup:**  
  We observe data from multiple environments $e \in \mathcal{E}$, each providing samples $(X^e, Y^e)$.  
  Distributions may differ across environments, but causal mechanisms should remain stable.
- **Core Assumption:**  
  There exists a subset $S^\* \subseteq \{1, \ldots, p\}$ such that  
  $$
  P(Y^e \mid X^e_{S^\*}) = P(Y^{e'} \mid X^{e'}_{S^\*}), \quad \forall e, e' \in \mathcal{E}.
  $$  
  Equivalently, in a linear model  
  $$
  Y^e = \mu + X^e \gamma^\* + \varepsilon^e, \quad \varepsilon^e \sim F_\varepsilon, \ \varepsilon^e \perp X^e_{S^\*},
  $$  
  where the noise distribution $F_\varepsilon$ is identical across environments.
- **Key Idea:**  
  The correct causal predictors $S^\*$ make the conditional distribution of $Y$ invariant under environment changes.  
  Any subset missing a true parent breaks this invariance.
- **Statistical Formulation:**  
  For every candidate set $S \subseteq \{1, \ldots, p\}$, test the null hypothesis  
  $$
  H_{0,S}: P(Y^e \mid X^e_S) \text{ is identical across all } e \in \mathcal{E}.
  $$  
  In linear settings, this corresponds to testing whether regression coefficients and residual distributions are stable across environments.
- **Algorithmic Procedure:**  
  1. For each subset $S$, fit regressions $Y^e \sim X^e_S$ for all environments.  
  2. Test invariance of coefficients and residuals across environments.  
  3. Keep all subsets where $H_{0,S}$ is not rejected.  
  4. Define the final causal set as  
     $$
     \hat{S} = \bigcap_{S:\,H_{0,S}\text{ not rejected}} S.
     $$
- **Guarantees:**  
  If tests are valid at level $\alpha$, ICP ensures  
  $$
  P(\hat{S} \subseteq S^\*) \ge 1 - \alpha,
  $$  
  meaning it will not include false causal variables.  
  Confidence intervals for coefficients $\gamma_j$ can also be constructed.
- **Intuitive Summary:**  
  ICP transforms the *philosophical principle* of causal invariance into a *testable statistical procedure*:  
  by searching for variable sets that make the conditional distribution of $Y$ stable across environments,  
  it identifies direct causes with statistical guarantees.
