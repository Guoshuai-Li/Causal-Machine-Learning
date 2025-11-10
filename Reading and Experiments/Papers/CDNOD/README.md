# CD-NOD 
---

## 1. Research Background and Motivation

In traditional causal discovery research, data typically comes from a **stationary distribution** or **homogeneous data**, that is:

P(V₁, V₂, ..., Vₙ) = ∏ᵢ P(Vᵢ | PAᵢ)

where each causal module P(Vᵢ | PAᵢ) remains unchanged across time or different environments.

However, in the real world—such as in **brain neural activity (fMRI)**, **climate change**, **economic systems**, and other domains—organisms/systems are in constant or varying conditions over time or experimental conditions. This **nonstationarity** or **heterogeneity** makes traditional methods (such as PC, LiNGAM, ANM) ineffective, manifesting as:

- Spurious **spurious independencies**
- **Direction ambiguity**
- Inconsistent model parameters across different environments

Zhang et al. proposed **CD-NOD (Constraint-based causal Discovery from Nonstationary / heterogeneous Data)**, using a framework to solve two core questions:

1. How to identify which variables' causal modules change over time or environment?
2. How to use distribution changes and direction to determine causal direction?

---

## 2. Problem Definition and Modeling Assumptions

### 2.1 Problem Setup

Given an observation variable set V = {V₁, ..., Vₙ}, the causal structure is represented by DAG G.

Assume there exists an additional variable C (time or domain index), with joint distribution written as:

P(V, C) = ∏ᵢ P(Vᵢ | PAᵢ, θᵢ(C))

Where:

- θᵢ(C): Model parameters related to environment C
- When θᵢ(C) changes with C, then Vᵢ's causal module is non-stationary (**changing module**)

### 2.2 Key Assumptions

- **Pseudo Causal Sufficiency (Pseudo Causal Sufficiency)**:
  If there exists a confounder (confounder), its influence can be viewed as C's marginal effect g(C), represented by C.
- **Independent but identically distributed samples**: Samples are independent under different C, but distributions differ.
- **Instantaneous causal relationships**: Only considers causal effects at the same time point, no temporal lag.

This setup allows some causal modules or exposure methods to change over time, while some causal structures remain stable across environments.

---

## 3. Phase 1 — Detecting Change Modules and Causal Skeleton Estimation

### 3.1 Basic Idea

Introduce C as a proxy variable, through conditional independence testing on V ∪ {C}:

- Identify variables related to C (changing modules)
- Simultaneously restore the undirected causal structure **skeleton**

The algorithm logic is as follows:

---

### Algorithm 1: CD-NOD Phase 1

**1. Construct complete undirected graph** UC, with node set as V ∪ {C}.

**2. Detect changing modules**:
   For each Vᵢ, if there exists a subset S ⊆ V \ {Vᵢ}, such that
   
   Vᵢ ⊥ C | S
   
   holds, then Vᵢ's module is confirmed; otherwise, if the module changes with C.

**3. Restore skeleton structure**:
   For each pair of variables Vᵢ, Vⱼ, if there exists a subset S ⊆ (V \ {Vᵢ, Vⱼ}) ∪ {C},
   satisfying
   
   Vᵢ ⊥ Vⱼ | S
   
   then remove edge Vᵢ — Vⱼ.

**Theorem 1 Guarantee**: Under causal sufficiency and faithfulness assumptions, this method can restore the true skeleton.

---

### 3.2 Implementation Details

- Use **kernel conditional independence test (KCI-test)** [Zhang et al., 2011] to capture non-linear dependencies
- Compared to PC algorithm using only variable independence, CD-NOD conducts C **simultaneous testing of time dependence + causal module drift**
- Identify change variables in results and C correlation, that is **"C-specific variables"** (nodes that change with time or environment)

---

## 4. Phase 2 — Non-stationarity Assists Causal Direction Determination

Non-stationarity is not just noise, but can serve as a "natural experiment" revealing causal direction.

CD-NOD's second stage uses the "**Independent Changes of Causal Modules (ICCM)**" principle for directionality.

---

### 4.1 Core Principle: Independent Changes

Under unconfounded conditions, causal system's different modules P(X) and P(Y|X) are independently changing:

if X → Y,   ΔP(X) ⊥ ΔP(Y|X)

Conversely, if Y → X, then the two changes are correlated.

**Extended explanation**:

Traditional ICP assumes causal mechanisms are invariant.

CD-NOD recognizes machines can change, but **changes are independent**,

This mechanism is more flexible and closer to non-stationary reality.

---

### 4.2 Causal Direction Determination Rules

Set variable Vₖ as C-specific change variable, compare its relationship with each neighbor Vₗ:

**Case 1: Vₗ is not related to C**

- If C — Vₖ — Vₗ forms a non-shielded triadic structure, then:
  - If C ⊥ Vₗ | S \ {Vₖ}, then V structure: Vₗ → Vₖ ← C
  - Otherwise Vₖ → Vₗ

**Case 2: Vₗ is also related to C**

- For two changing modules, compare their independent changes:
  - If P(Vₗ) and P(V₂|Vₗ) change independently, then determine Vₗ → V₂
  - Otherwise, then V₂ → Vₗ

In implementation, approximate verification of change independence through statistical estimation based on degree of mutual dependence.

---

## 5. Experimental Results and Validation

### 5.1 Synthetic Data

In 6 node non-stationary SFM models, CD-NOD compared with PC/SGS methods:

- **False positive (FP)** rate significantly decreased
- Able to correctly identify time-varying errors
- **Causal direction determination accuracy reaches 93%**
- When sample size is sufficient (N=1000), **FN** rate approaches stable

---

### 5.2 Real Applications

**(1) fMRI Hippocampus Dataset**

- **Data**: 64 days, 6 brain regions signals
- CD-NOD reduced FP from 62.9% to 17.1%
- **Stable edges**: CA3 → CA1, CA1 → Sub, Sub → ERC
- **Direction determination accuracy** ≈ 85.7%

**(2) Breast Tumor Dataset (UCI)**

- Identified 11 tumor-type-related features
- Using these features for SVM classification, CV error reduced to 0.0246
- Demonstrates CD-NOD helps discover stable, interpretable feature sets

---

## 6. Relationship with Granger / ICP / PCMCI

| Method | Research Object | Mechanism Assumption | Handling Distribution Changes | Causal Direction Principle | Typical Application |
|--------|----------------|---------------------|------------------------------|---------------------------|---------------------|
| **Granger (1969)** | Stationary time series | Fixed mechanism | Not supported | Prediction improvement | Economics, neural signals |
| **ICP (2016)** | Multi-environment data | Mechanism invariance | Supports (invariance assumption) | Stable prediction | Distribution differentiation |
| **PCMCI (2019)** | High-dimensional time series | Local independence | Detectable time lag | Conditional independence | Climate/neural systems |
| **CD-NOD (2017)** | Non-stationary or heterogeneous data | Module can change but is independent | Strong support | Change independent | fMRI, distribution migration scenarios |

**Summary**:

CD-NOD inherits ICP's "invariance assumption" and PCMCI's "time series structure discovery",

Using distribution change information, it "upgrades" causal discovery to "causal module independent changes".

---

## 7. Method Contributions and Limitations

###  Contributions

1. Provides a unified framework for handling non-stationary / heterogeneous data causal discovery
2. First systematic proposal of **"Independent Changes Principle (ICCM)"**
3. Uses distribution changes to naturally reverse information for causal direction inference
4. Provides non-parametric, core method implementation (**KCI-test**), suitable for complex non-linear relationships
5. Validates effectiveness in multiple domain real data

###  Limitations

- Still assumes DAG does not change over time (only mechanism changes)
- If confounding influence is strong or non-stationary heterogeneity is too complex, ICCM may fail
- Sample size, time, independent testing success rate affects tradeoff

---

## 8. Core Insights

CD-NOD's fundamental insight is:

**Distribution change is not noise, but a causal discovery signal.**

By letting "time" or "environment" participate in causal learning,

CD-NOD achieves from "static causality" to "dynamic mechanism" expansion,

providing theoretical bridges for modern causal inference and distribution generalization.
