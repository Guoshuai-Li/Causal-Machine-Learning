# Week 5: Causal Structure Learning

## Overview
This experiment explores computational methods for discovering causal structures from observational data, comparing constraint-based and score-based approaches while examining their fundamental limitations and assumptions.

## Learning Objectives
- Implement PC algorithm for conditional independence-based structure discovery
- Apply GES algorithm using BIC scoring for causal graph search
- Understand Markov equivalence classes and structural identifiability limits
- Analyze faithfulness assumption violations and algorithm failure modes

## Experimental Setup

### True DAG Structure
```
Generated star-shaped DAG:
X0 → X1 → {X2, X3, X4}

Edges: [('X0', 'X1'), ('X1', 'X2'), ('X1', 'X3'), ('X1', 'X4')]
Variables: 5, Edges: 4, Sample size: 1000
```

### Key Questions
1. **Constraint vs Score Methods**: How do PC and GES algorithms compare in structure recovery?
2. **Markov Equivalence**: Which causal structures are fundamentally indistinguishable from data?
3. **Faithfulness Violations**: When do standard assumptions fail and algorithms break down?
4. **Sample Complexity**: How does data size affect structure learning performance?

## Results Summary

### Algorithm Performance Comparison
- **PC Algorithm**: Precision=0.500, Recall=0.500, F1=0.500, SHD=4
  - Learned: [X1→X0, X1→X4, X1→X3, X2→X1] (2/4 edges correct)
  - Issue: Edge orientation errors due to simplified heuristics
- **GES Algorithm**: Precision=1.000, Recall=1.000, F1=1.000, SHD=0
  - Learned: [X0→X1, X1→X2, X1→X4, X1→X3] (perfect recovery)
  - Success: Global BIC optimization overcame local independence test limitations

### Markov Equivalence Verification
- **Chain A→B→C**: A⊥C|B = True (equivalent structures)
- **Fork A←B→C**: A⊥C|B = True (equivalent structures)
- **Collider A→B←C**: A⊥C|B = False (distinguishable structure)

### Faithfulness Assumption Test
- **Marginal independence**: X⊥Y = True (correlation = -0.004)
- **Conditional dependence**: X⊥Y|Z = False
- **Parameter cancellation**: X→Z(+0.698) vs Y→Z(-0.716) effects nearly cancel
- **Algorithm impact**: Violates faithfulness, causing PC algorithm failures

### Sample Size Effects
- **Consistent PC performance**: Precision/Recall stable at 0.500 across n∈{100,500,1000,2000}
- **No improvement trend**: Additional data did not resolve edge orientation issues
- **Method limitation**: Systematic bias in simplified PC implementation

## Key Findings
1. **Score-based superiority**: GES dramatically outperformed PC on this star structure (100% vs 50% accuracy)
2. **Equivalence class reality**: Chain and fork structures genuinely indistinguishable from conditional independence patterns
3. **Faithfulness fragility**: Carefully constructed parameter cancellation successfully broke standard assumptions
4. **Sample complexity**: More data insufficient to overcome fundamental algorithmic limitations

## Files
- `structure_learning.py` - Main experiment code
- `experiment_results.png` - CPDAG visualization
- `analysis.md` - Detailed findings and interpretation

## How to Run
```bash
python structure_learning.py
```

**Requirements**: numpy, pandas, scikit-learn, matplotlib, networkx, scipy
