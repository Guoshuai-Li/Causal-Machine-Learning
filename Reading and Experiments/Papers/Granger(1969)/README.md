# Granger Causality 
---

## 1. Research Background and Problem Motivation

In the 1960s, economists faced a core dilemma:

In time series data, when two variables are highly correlated, **how to determine which is the cause and which is the effect**?

For example, is the high correlation between loan supply and price levels caused by monetary policy driving price changes, or is it the reverse?

The prevailing **Simultaneous Equation Models** often assumed variables existed in **instantaneous causality**, but lacked a verifiable directional definition.

Granger's work first proposed a **testable** definition of causality **based on time**, which laid the foundation for statistical causal inference.

His core idea is:

> If the past of $Y$ contains independent predictive information about the future of $X$ (i.e., can reduce prediction error),
> then **Y "Granger causes" X**.

---

## 2. Core Ideas and Definition

Granger transforms **causal relations** into a **testable prediction improvement problem**.

By comparing "the prediction ability using the past of $X$ alone" versus "jointly using the past of $X$ and the past of $Y$" to determine causality.

Setup:

- $X_t, Y_t$: stationary time series;
- $\varepsilon_t(X \mid Y)$: prediction error;
- $\sigma^2(X \mid Y)$: error variance.

### 2.1 Causality Definition

If:

$$\sigma^2(X_t \mid X_{t-1}, X_{t-2}, \ldots, Y_{t-1}, Y_{t-2}, \ldots) < \sigma^2(X_t \mid X_{t-1}, X_{t-2}, \ldots),$$

then **Y Granger-causes X**, denoted as $Y_t \Rightarrow X_t$.

That is: given the information of $X$ itself from the past, introducing the past of $Y$ can improve the prediction of $X$, then $Y$ has a causal effect on $X$.

### 2.2 Feedback

If both:

$$Y_t \Rightarrow X_t \quad \text{and} \quad X_t \Rightarrow Y_t,$$

then there exists **bidirectional feedback**.

### 2.3 Instantaneous Causality

If the current value of $Y_t$ can improve the prediction of $X_t$ (beyond past information), then there exists instantaneous causality:

$$\sigma^2(X_t \mid X_{t-1}, Y_t) < \sigma^2(X_t \mid X_{t-1}).$$

Granger pointed out that **causality is determined by sample frequency and is not absolute** — what appears as instantaneous correlation at one timescale may actually be revealed as "true time sequence" at a finer temporal resolution.

---

## 3. Model Derivation and Formalization Description

### 3.1 Two-variable VAR Form

Consider a bivariate stationarity process:

$$
\begin{cases}
X_t = \sum_{j=1}^m a_j X_{t-j} + \sum_{j=1}^m b_j Y_{t-j} + \varepsilon_t, \\
Y_t = \sum_{j=1}^m c_j X_{t-j} + \sum_{j=1}^m d_j Y_{t-j} + \eta_t,
\end{cases}
$$

where $\varepsilon_t, \eta_t$ are white noise.

- If $b_j \neq 0$, then $Y \Rightarrow X$;
- If $c_j \neq 0$, then $X \Rightarrow Y$;
- If both are non-zero, then there is feedback.

This is the modern **Vector Autoregression (VAR)** model form.

---

### 3.2 Expanded Form of Instantaneous Causality

Allowing for contemporaneous terms:

$$
\begin{cases}
X_t + b_0 Y_t = \sum_{j=1}^m a_j X_{t-j} + \sum_{j=1}^m b_j Y_{t-j} + \varepsilon_t, \\
Y_t + c_0 X_t = \sum_{j=1}^m c_j X_{t-j} + \sum_{j=1}^m d_j Y_{t-j} + \eta_t.
\end{cases}
$$

If $b_0, c_0 \neq 0$, then there exists instantaneous causality.

---

### 3.3 Testable Form (Linear Prediction Difference Comparison)

Based on the prediction error setting, the construction test hypotheses:

$$H_0: b_1 = b_2 = \cdots = b_m = 0 \quad \text{vs.} \quad H_1: \exists b_j \neq 0.$$

In samples, this can be achieved through F-test or Wald test.

This idea became the basis for the **Granger causality test** (statistical foundation established by Sims and others in the 1970s for standard testing framework).

---

## 4. Spectral Analysis Perspective (Spectral and Cross-spectral Methods)

To study causal relations in different frequency ranges, Granger further proposed **frequency domain** analysis.

### 4.1 Core Idea

If $X_t, Y_t$ are stationary processes, they can be represented using Fourier time expressions:

$$X_t = \int_{-\pi}^{\pi} e^{i\omega t} dZ_X(\omega), \quad Y_t = \int_{-\pi}^{\pi} e^{i\omega t} dZ_Y(\omega).$$

Define:

- **Power spectrum** $f_X(\omega)$: degree of variance of each frequency component;
- **Cross-spectrum** $C_{XY}(\omega)$: degree of co-movement between $X$ and $Y$ that is not synchronized on different frequency rates.

### 4.2 Causal Decomposition

Granger proved that in a bidirectional feedback system,

$$C_{XY}(\omega) = C_1(\omega) + C_2(\omega),$$

where:

- $C_1(\omega)$: caused by $X \Rightarrow Y$;
- $C_2(\omega)$: caused by $Y \Rightarrow X$.

This allows determining **causal strength and direction** in the frequency domain.

### 4.3 Causal Coherence and Relative Position

Define causal coherence:

$$C_{XY}^{(1)}(\omega) = \frac{|C_1(\omega)|^2}{f_X(\omega)f_Y(\omega)},$$

and relative phase:

$$\phi_{XY}(\omega) = \tan^{-1}\left(\frac{\text{Im}[C_1(\omega)]}{\text{Re}[C_1(\omega)]}\right),$$

which can obtain **lag** and **strength at different frequency rates** used in causal analysis.

---

## 5. Theoretical Significance and Philosophical Interpretation

### 5.1 Causality Definition Based on Time Irreversibility

Granger clearly proposed:

> **"The future cannot cause the past"**
> *(the future cannot cause the past)*

Causal relationships are thus rooted in the **directionality of time series**, not purely statistical correlation.

### 5.2 Transformation from Correlation to Causality

Unlike earlier structural equation methods, Granger causality:

- **Does not depend on structural assumptions**;
- **Can be manipulated**;
- **Can be verified through prediction accuracy**.

It transforms the **philosophical problem of causal relations** into a **statistical problem of predictive performance**.

---

## 6. Connections with Modern Causal Learning

| Feature | Granger (1969) | ICP | Anchor Regression | IRM |
|---------|----------------|-----|-------------------|-----|
| Core Problem | Time series causal directionality determination | Multi-environment causal invariance | Balance return and causal extrapolation | Cross-domain invariant risk minimization |
| Theoretical Foundation | Prediction improvement + time directionality | Component distribution invariance | Distribution robustness | Mechanism stability |
| Data Type | Dynamic time series | Multi-environment static data | Heterogeneous data with anchors | Multi-task/multi-domain data |
| Testability | High: F/Wald test | Medium: statistical hypothesis testing | Continuous γ-path | Relies on constraint optimization |
| Causal Hypothesis | "Past → Future" irreversible | Causal mechanism invariance | Descriptive external properties | No need to show causal structure decomposition |
| Philosophical Direction | Causality = improved prediction | Causality = invariance mechanism | Causality = robust parameters | Causality = invariant risk |

---

## 7. Contributions and Limitations

### 7.1 Main Contributions

1. **First proposed a testable causality definition**, laying the mathematical foundation for "Granger causality";
2. **Linked prediction accuracy with causal direction** in a directly verifiable way;
3. **Combined frequency domain analysis**, proposing causal intensity and lag measurement;
4. **Laid the foundation for later VAR models**, providing the basis for modern macroeconomic dynamic building and modern time series causal inference.

### 7.2 Limitations and Critiques

- **Assumes time series stationarity**, difficult to handle structural changes;
- **Only for linear causality**, cannot handle nonlinear dynamics;
- **Immediate causality is difficult to accurately interpret** in relation to sample frequency dependency;
- **Not the same as non-linear Granger causality**, transfer entropy, etc.

---

## 8. Core Insights and Summary

> **Granger** causality is "prediction improvement → causal interpretation" bridge.
> It first provided a verifiable approach, incorporating time into causal reasoning.
> It promoted academia from statistically symmetric correlation to direction-focused causal structural modeling and supply theoretical foundations.

**Core Insight:**

> "If the past of a variable genuinely does not improve another variable's future prediction, it is not causal in meaning." —— This defines a measurement framework enabling modern causal learning development and theoretical foundations.

---
