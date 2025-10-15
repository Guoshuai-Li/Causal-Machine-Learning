## Question: Why is the empty set S mentioned? And what is the meaning of "invariance" in the context of "empty set"?

**Answer:** 
- When $S = \emptyset$, it means we have not released any cause to change.
- If $Y$'s distribution has already changed under different experimental conditions, then we obviously cannot say that $Y$'s distribution is "invariant".
- Therefore, we will conclude that "the empty set is not invariant".
- At this point, we need to find some $X_s$ (i.e., add and test variables) to see if we can solve this environmental change problem.
- Once we find an $S$ such that the distribution of $Y|X_S$ is invariant across environments, we can recognize this $S$ as a causal structure.

In summary, rejecting the invariance assumption for the empty set $S = \emptyset$ means that $Y$ has inconsistent marginal distributions across different environments. Since $S = \emptyset$ represents "not releasing any cause", this invariance rejection requires that $Y$ itself has different distributions across all environments.

---
## Question: Why use a weakened assumption?

**Answer:**

The original null hypothesis $H_{0,S}(\mathcal{E})$ (Equation 10) is too strict and difficult to verify:

**Original Hypothesis Requirements:**
- The residual distribution must be completely identical across all environments
- All distributions need to be tested for identical shape
- The independence condition $\varepsilon^e \perp X_S^e$ cannot be verified in finite samples

Therefore, they directly use $H_{0,S}$ for statistical testing, which is difficult to control the Type I error rate.

**Solution: Weakened Assumption** (Definition $\tilde{H}_{0,S}(\mathcal{E})$, Equation 16):

They retained the core and testable parts:
- Cross-environment "linear structure" and "variance level" are consistent

Specifically, the weakened version only requires:
- "The optimal regression coefficient across environments $\beta^{\text{pred},e}(S)$ is consistent"
- "The residual variance $\sigma^e(S)$ is consistent"

**Why is it called "weakened"?**
- If the original hypothesis $H_{0,S}(\mathcal{E})$ holds, then the weakened version $\tilde{H}_{0,S}(\mathcal{E})$ must also hold
- However, the converse is not necessarily true, because $\tilde{H}_{0,S}$ does not require the residual distribution to be completely identical
- Therefore, it is a more relaxed (easier to satisfy) assumption

**Purpose of this approach:**

The main purpose is to construct an effective statistical test.

Based on this weakened version, they can use standard statistical tests (such as F-test / Chow test) to verify:
- Whether the regression coefficients are consistent across different environments
- Whether the residual variances are consistent across environments

These can be tested using classical distributions (F-distribution), providing a well-calibrated significance level $\alpha$ control.

Therefore, the result is:

$$\hat{\Gamma}_S(\mathcal{E}) \rightarrow \begin{cases} \emptyset, & \text{if } \tilde{H}_{0,S}(\mathcal{E}) \text{ rejected at level } \alpha, \\ \hat{C}(S), & \text{otherwise.} \end{cases}$$

---
