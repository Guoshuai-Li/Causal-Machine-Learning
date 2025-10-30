# Invariant Causal Prediction (ICP) — 10-Minute Talk Version

---

## 1. Why We Care

Let's start with a simple question:

Why do machine learning models often fail when we deploy them in the real world?

It's because what they learn are **correlations**, not **causal relationships**.

As soon as the data distribution shifts — maybe the policy changes, or we run a new experiment — those correlations fall apart.

So, what we really want is to find the variables that have a **stable and causal** relationship with our target $Y$.

And this is exactly what *Invariant Causal Prediction*, or ICP, is designed to do.

---

## 2. The Core Idea — Stability as a Signal of Causality

The central principle behind ICP is very intuitive:

> if a relationship holds true under change, it's probably causal.

In other words, if $Y$ truly depends on a certain set of variables $S^\*$,

then the way $Y$ relates to those variables should **stay the same** across different environments.

Formally, that means the conditional distribution $P(Y \mid X_{S^\*})$

doesn't change when we move from one environment $e$ to another.

So instead of trying to recover an entire causal graph,

ICP focuses on one target $Y$ and asks:

"Which predictors make the behavior of $Y$ stable across environments?"

---

## 3. How It Works — The Statistical Engine

Here's the high-level process.

We assume we have data from several environments — maybe different labs, time periods, or intervention settings.

For each candidate subset of predictors $S$, we check whether the relationship between $X_S$ and $Y$ is **invariant** across these environments.

Concretely, we test a hypothesis:

> "Does $P(Y^e \mid X_S^e)$ look the same in all environments?"

If yes, we keep $S$; if not, we discard it.

Finally, we take the **intersection** of all the sets that passed the test.

This gives us a subset of variables that are most likely the **direct causes** of $Y$.

And here's the beauty: ICP also gives **statistical guarantees**.

If our tests are run at significance level $\alpha$,

we can be confident that the discovered set $\hat{S}$

is contained within the true causal set $S^\*$ with probability at least $1 - \alpha$.

So it's not just heuristic discovery — it's causal inference with error control.

---

## 4. Where It Sits in the Causal Landscape

Now, how is this different from all those other causal discovery methods we know — PC, GES, LiNGAM, and so on?

Most of them rely on **conditional independence** assumptions or **specific functional forms**, and they usually aim to reconstruct the *entire causal DAG*.

ICP takes a different route.

It doesn't need faithfulness assumptions or knowledge about which variable was intervened on.

All it needs to know is: "These datasets come from different environments."

So you can think of ICP as **a lightweight, target-focused causal discovery tool**.

It's also **model-agnostic** — it can work with linear or nonlinear models —

and it produces **confidence statements**, which most discovery methods can't.

In short:

> ICP shifts the question from "What's the full causal structure?"
> to "Which predictors are truly stable causes of my target variable?"

---

## 5. Why It Matters

This stability perspective is powerful far beyond pure causality.

It connects directly to **robust machine learning** and **out-of-distribution generalization**.

In a way, ICP gives a theoretical foundation for learning models that *don't break* when the world changes.

Practically, it has been applied to things like gene perturbation studies, economic modeling, and domain adaptation in ML.

And theoretically, it was one of the first methods to bring **confidence intervals** into causal discovery.

So when we say *Invariant Causal Prediction*,

we're really talking about bridging **causal reasoning** and **statistical reliability** —

a combination that's increasingly important in modern AI research.

---

## 6. The Takeaway

To wrap up:

> ICP reframes causal discovery as testing for **stability across environments**.
> If a relationship remains invariant under change, it's a strong signal of causality.
> By turning this principle into a statistical test, ICP lets us identify causal predictors of $Y$
> — with confidence guarantees and without needing to know where the interventions happened.

That's why ICP isn't just another causal discovery algorithm.

It's a **way of thinking** about causality —

one that builds bridges between theory, robustness, and real-world learning.
