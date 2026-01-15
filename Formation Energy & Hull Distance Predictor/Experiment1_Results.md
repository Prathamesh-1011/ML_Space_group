# Experiment 1: Formation Energy & Hull Distance Prediction (Composition-Only)

## Objective

The goal of this experiment was to develop a **composition-only machine learning model** to predict:

1. **Formation Energy per atom (regression)**
2. **Energy Above Hull / Hull Distance (regression)**
3. **Thermodynamic Stability (binary classification)**

This model is intended as a **pre-screening filter** to eliminate unstable compounds before expensive DFT calculations.

---

## Dataset Construction

* **Source:** Materials Project database (via MP API)
* **Input data type:** Composition-only (chemical formula, no crystal structure)
* **Raw properties retrieved:**

  * Formation energy per atom
  * Energy above hull
* **Derived label:**

  * Stability label defined as:

    * `Stable = 1` if hull distance ≤ 0.05 eV/atom
    * `Unstable = 0` otherwise

### Dataset Characteristics

* Only **lowest-energy polymorphs** per composition were retained
* Missing or invalid entries were removed
* Final dataset was stored and processed in **CSV format**

---

## Feature Engineering

* Chemical formulas were converted into **numerical representations** suitable for ML
* Features capture **stoichiometric and elemental information only**
* No structural, symmetry, or graph-based information was used

This ensures the model is applicable at the **early discovery stage**, before crystal structures are known.

---

## Model Architecture

### Multi-Task Learning Setup

A **single shared model** was trained with three outputs:

1. **Formation Energy Head** (regression)
2. **Hull Distance Head** (regression)
3. **Stability Head** (binary classification)

Shared layers learn common chemical representations, while task-specific heads specialize for each target.

---

## Training Strategy

* Dataset split into **training and test sets**
* Loss functions:

  * **MAE loss** for formation energy
  * **MAE loss** for hull distance
  * **Binary cross-entropy** for stability classification
* Joint optimization using weighted multi-task loss
* Standard normalization applied to numerical targets
* Model trained end-to-end until convergence

---

## Evaluation Metrics

The following metrics were used:

| Task             | Metric                    |
| ---------------- | ------------------------- |
| Formation Energy | Mean Absolute Error (MAE) |
| Hull Distance    | Mean Absolute Error (MAE) |
| Stability        | F1-score                  |

---

## Results

### Final Test Performance

```
Formation Energy MAE: 0.117314905 eV/atom
Hull Distance MAE:    0.071798556 eV/atom
Stability F1-score:   0.902604808
```

---

## Result Interpretation

* **Formation Energy Prediction**

  * Achieved moderate accuracy for a composition-only model
  * Higher error compared to state-of-the-art deep models (e.g., CrabNet/Roost), but suitable for coarse filtering

* **Hull Distance Prediction**

  * Performance falls within the expected range reported for composition-only approaches
  * Effective for identifying near-hull candidates

* **Stability Classification**

  * Very high F1-score indicates strong separation between stable and unstable compounds
  * Demonstrates effectiveness as a **binary screening filter**

---

## Key Outcomes

* Successfully built a **composition-only, multi-task screening model**
* Demonstrated that:

  * Regression + classification can be learned jointly
  * Stability prediction is highly reliable even when regression errors are moderate
* The model is suitable for:

  * High-throughput pre-screening
  * Reducing downstream DFT cost
  * Acting as **Stage-1 filtering** in a materials discovery pipeline

---

## Limitations Observed

* Formation energy accuracy is lower than attention-based or message-passing models
* No uncertainty estimation implemented in this experiment
* Interpretability limited due to lack of explicit attention visualization

---

## Experiment Status

✅ **Completed successfully**
📌 Serves as **baseline experiment** for future improvements:

* Ensemble learning
* Learning rate scheduling & early stopping
* Attention-based interpretability
* Roost/CrabNet-style architectures

---