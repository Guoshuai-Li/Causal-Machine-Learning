\section*{Invariant Causal Prediction (ICP)}

\begin{itemize}
    \item \textbf{Problem:} Models often fail under distribution shifts or interventions; correlations learned from one environment may break in another.
    
    \item \textbf{Goal:} Identify variables whose relationship with the target \textit{remains invariant} across multiple environments.
    
    \item \textbf{Key Assumption:} The conditional distribution $P(Y \mid X_{S^*})$ \textit{stays the same under all environments} if $S^*$ contains the true causal parents of $Y$.
    
    \item \textbf{Approach:} Test each candidate set $S \subseteq \{X_1, \ldots, X_p\}$ for invariance of $P(Y \mid X_S)$ across environments, and take the intersection of accepted sets.
    
    \item \textbf{Outcome:} A statistically valid subset $\hat{S} \subseteq S^*$ of direct causal predictors and confidence intervals for their effects.
    
    \item \textbf{Intuition:} Causal mechanisms are stable --- exploiting this stability turns causality into a testable statistical property.
\end{itemize}
