# Experiment Analysis

## Background
This experiment tests computational causal discovery methods using a randomly generated 5-variable star-shaped DAG where X1 serves as a central hub connected to all other variables, representing a challenging structure for edge orientation algorithms.

## Results Interpretation

### 1. PC Algorithm Performance Analysis
**What we tested**: Constraint-based structure learning using conditional independence tests and simplified edge orientation.

**Results**:
- Learned structure: [X1→X0, X1→X4, X1→X3, X2→X1]
- True structure: [X0→X1, X1→X2, X1→X3, X1→X4]
- Performance: Precision=0.500, Recall=0.500, F1=0.500, SHD=4
- **Edge detection success but orientation failures**

**Why PC struggled**: The algorithm correctly identified the skeleton (which variables are connected) but failed at edge orientation. Our simplified orientation heuristic (based on marginal variances) proved inadequate for the star structure where X1 has multiple parents and children, making variance-based direction assignment unreliable.

### 2. GES Algorithm Superior Performance
**What we tested**: Score-based structure learning using BIC optimization with greedy search.

**Results**:
- Learned structure: [X0→X1, X1→X2, X1→X4, X1→X3] (perfect match)
- Performance: Precision=1.000, Recall=1.000, F1=1.000, SHD=0
- **Complete structure recovery achieved**

**Why GES succeeded**: The global BIC scoring approach evaluated entire graph structures rather than local independence relationships. The forward-backward search strategy effectively navigated the space of possible DAGs, and the star structure's clear directional pattern (single hub) was well-captured by BIC's preference for simpler explanations.

### 3. Markov Equivalence Class Verification
**What we tested**: Whether different causal structures can have identical conditional independence signatures.

**Results**:
- Chain A→B→C: A⊥C|B = True
- Fork A←B→C: A⊥C|B = True  
- Collider A→B←C: A⊥C|B = False
- **First two structures are genuinely indistinguishable from observational data**

**Why equivalence exists**: Chain and fork structures both imply that A and C are independent given B, making them statistically equivalent despite different causal interpretations. Only the collider structure breaks this pattern because conditioning on B creates dependence between A and C through selection bias.

### 4. Faithfulness Assumption Violation
**What we tested**: Construction of a scenario where causal connections exist but appear statistically independent due to parameter cancellation.

**Results**:
- Marginal correlation X-Y: -0.004 (effectively zero)
- X→Z correlation: +0.698, Y→Z correlation: -0.716
- X⊥Y|Z test result: False (correctly detects dependence given Z)
- **Successfully violated faithfulness assumption**

**Why violation occurred**: We engineered X→Z and Y→Z effects with nearly equal magnitudes but opposite signs (+0.698 vs -0.716), causing their marginal correlation through Z to cancel out. This creates a scenario where X and Y appear independent marginally but are dependent given Z, violating the faithfulness assumption that all causal relationships manifest as statistical dependencies.

### 5. Sample Size Effects Analysis
**What we tested**: Whether increasing sample size improves PC algorithm performance on the star structure.

**Results**:
- All sample sizes (100, 500, 1000, 2000): Precision/Recall stable at 0.500
- No improvement trend observed with additional data
- **Systematic algorithmic limitation rather than sample size issue**

**Why sample size didn't help**: The PC algorithm's poor performance stemmed from our simplified edge orientation heuristic, not from insufficient statistical power for independence testing. More data cannot overcome fundamental algorithmic limitations in orientation logic.

## Visual Analysis

### CPDAG Structure Interpretation
- Central hub X1 clearly visible with connections to all other variables
- Star pattern well-represented in the spring layout visualization
- Graph structure enables clear understanding of the learning challenge

## Practical Implications

### 1. Method Selection Guidelines
- **PC algorithm**: Better for skeleton discovery, requires sophisticated orientation rules
- **GES algorithm**: Superior for complete structure recovery when BIC assumptions met
- **Structure dependency**: Star/hub structures particularly challenging for constraint-based methods

### 2. Real-World Considerations
- **Equivalence classes**: Many causal questions fundamentally unanswerable from observational data alone
- **Faithfulness fragility**: Parameter tuning or measurement precision can break standard assumptions
- **Algorithm choice**: Score-based methods may be more robust for complex structures

### 3. Validation Strategies
- **Multiple algorithms**: Run both constraint and score-based methods for comparison
- **Assumption testing**: Check faithfulness violations through residual analysis
- **Domain knowledge**: Incorporate expert knowledge to resolve orientation ambiguities

## Methodological Lessons

### What Worked Well
- **GES robustness**: Excellent performance on moderately complex structure
- **Equivalence demonstration**: Clear illustration of fundamental identifiability limits
- **Faithfulness testing**: Successful construction of assumption-violating scenarios

### What Revealed Limitations
- **PC orientation**: Simplified heuristics inadequate for complex structures
- **Sample complexity**: More data cannot solve algorithmic design issues
- **Method specificity**: Algorithm performance highly dependent on structure type

## Limitations

1. **Simplified implementations**: Full PC and GES algorithms have more sophisticated components
2. **Linear relationships**: Real causal relationships often non-linear
3. **Single structure type**: Results may not generalize to other DAG topologies
4. **Limited variables**: Scalability to high-dimensional problems unclear
5. **Perfect data**: No missing values or measurement error considered

## Bottom Line

**Score-based methods demonstrate clear advantages**: GES achieved perfect structure recovery while PC struggled with edge orientation, highlighting the value of global optimization over local conditional independence testing.

**Equivalence classes pose fundamental limits**: Markov equivalence is not just a theoretical concept but a practical constraint that limits what causal structures can be learned from observational data alone.

**Faithfulness violations are constructible**: The assumption that causal relationships manifest as statistical dependencies can be systematically broken through parameter cancellation, causing algorithm failures.

This experiment demonstrates both the promise and limitations of computational causal discovery, emphasizing the need for careful algorithm selection, assumption validation, and realistic expectations about what can be learned from data alone.
