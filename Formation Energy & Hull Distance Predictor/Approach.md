# Model-2: Formation Energy & Hull Distance Prediction  

## 1. Problem Framing

We formulate **Model-2** as a **multi-task supervised learning problem** using **composition-only data**.

### Inputs (from CSV)
- Chemical composition (formula)
- Stoichiometric fractions of elements

### Outputs (Targets)
1. **Formation Energy per atom** (continuous regression)
2. **Energy Above Hull (Ehull)** (continuous regression)
3. **Stability Label** (binary classification derived from Ehull)

This design mirrors the setup used in **CrabNet** and **Roost**, which are state-of-the-art for structure-agnostic thermodynamic prediction.


## 2. Why Multi-Task Learning (MTL)

Formation energy and hull distance are **thermodynamically coupled**:

- Formation energy determines relative phase stability
- Hull distance is computed from formation energy relative to competing phases

Training a **single shared model** to predict both properties:
- Improves generalization
- Reduces error via shared physical representations
- Matches best practices reported in Matbench Discovery and CrabNet


## 3. Input Representation (From CSV → Model)

### 3.1 Composition Encoding

Each material is represented as:
- A **set of elements**
- Associated **fractional stoichiometries**

Instead of handcrafted descriptors:
- Element identities are mapped to **learned embeddings**
- Stoichiometric fractions are **explicitly encoded**
- No structural or crystallographic information is used

This ensures:
- Invariance to formula ordering
- Compatibility with arbitrary chemistries
- No information leakage from structure


## 4. Shared Representation Learning

A **shared neural backbone** is trained to learn a latent representation of composition that captures:

- Elemental interactions
- Chemical trends (periodicity, electronegativity contrast)
- Composition complexity

This shared representation acts as a **thermodynamic fingerprint** of the material.


## 5. Task-Specific Prediction Heads

From the shared representation, three task-specific heads are attached:

### 5.1 Formation Energy Head
- Regression task
- Learns the absolute thermodynamic stability relative to elemental references
- Optimized using MAE or robust loss

### 5.2 Energy Above Hull Head
- Regression task
- Learns distance from the convex hull
- Benefits directly from shared information learned for formation energy

### 5.3 Stability Classification Head
- Binary classification
- Predicts whether Ehull < stability threshold
- Sharpens decision boundaries for screening


## 6. Training Strategy

### 6.1 Joint Optimization
- All tasks are trained **simultaneously**
- Losses are combined using weighted sum

This encourages:
- Physically consistent predictions
- Reduced overfitting to any single target

### 6.2 Data Splitting
- Train/validation/test splits are done **by composition**
- Polymorph leakage is avoided
- Matches Materials Project and Matbench protocols


## 7. Prediction Flow (CSV → Output)

1. Read composition and stoichiometry from CSV
2. Convert to model-compatible composition encoding
3. Pass through shared representation network
4. Predict:
   - Formation energy (regression)
   - Hull distance (regression)
   - Stability probability (classification)
5. Use predictions for:
   - Ranking candidate materials
   - Filtering unstable compositions
   - Prioritizing DFT calculations


## 8. Evaluation Metrics

### Regression Tasks
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Classification Task
- F1-score
- Precision–Recall balance (important for screening)

Evaluation focuses on **screening quality**, not just numerical accuracy.


## 9. Why This Approach Works with CSV Data

- Requires only composition → scalable to millions of candidates
- Avoids structural bias
- Matches the highest-accuracy regime achievable without structure
- Proven effective by CrabNet, Roost, and Matbench Discovery


## 10. Role in the Overall Discovery Pipeline

This model acts as a **pre-DFT filter**:
- Eliminates unstable candidates early
- Reduces DFT workload by orders of magnitude
- Feeds only high-probability stable materials into later CSP or structure-based models


## 11. Key Assumptions & Limitations

- Absolute stability cannot be guaranteed without structure
- Model predicts **thermodynamic likelihood**, not exact phase
- Designed for screening, not final validation

These limitations are explicitly acknowledged in the literature and are acceptable for early-stage discovery.


## Summary

From a CSV containing only composition and thermodynamic targets, we:
- Learn a shared chemical representation
- Predict formation energy and hull distance jointly
- Use the outputs for high-throughput stability screening

