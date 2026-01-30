# **Method 6: Uncertainty-Aware Ensemble Multi-Task Composition Model**

## 1. Method Overview

**Method 6** extends the previously developed composition-only multi-task learning framework by introducing **ensemble-based uncertainty estimation** and **risk-controlled prediction**.

Unlike Methods 1–5, which focus primarily on point prediction accuracy, Method 6 explicitly addresses **prediction reliability**, enabling the model to quantify *when its predictions should be trusted or rejected*.

This method is designed for **early-stage materials pre-screening**, where minimizing false confidence is as critical as minimizing error.


## 2. Relationship to Previous Five Methods

### Does Method 6 use a baseline from earlier methods?

**Yes.**

Method 6 **directly builds upon the baseline architecture used in Methods 1–5**, without introducing a new neural architecture.

| Aspect                    | Methods 1–5                 | Method 6               |
| ------------------------- | --------------------------- | ---------------------- |
| Input representation      | Composition only            | Same                   |
| Model architecture        | MLP with element embeddings | Same                   |
| Multi-task learning       | FE + Hull + Stability       | Same                   |
| Training strategy         | Single model                | **Ensemble of models** |
| Uncertainty estimation    | ❌                           | **✔ Explicit**         |
| Risk-controlled screening | ❌                           | **✔ Explicit**         |

### Key distinction

> **Methods 1–5 predict values.
> Method 6 predicts values *and* quantifies confidence.**

Thus, Method 6 should be viewed as a **decision-aware extension** of the earlier methods rather than a competing architecture.


## 3. Step-by-Step Methodology

### Step 1: Composition Encoding

Each material is represented using:

* Element indices (`elements`)
* Corresponding normalized atomic fractions (`fractions`)

Element embeddings are learned and aggregated using fraction-weighted summation to produce a fixed-length composition vector.


### Step 2: Shared Multi-Task Backbone

The aggregated composition vector is passed through a shared fully connected network that learns thermodynamic representations common to all tasks.


### Step 3: Multi-Task Outputs

Three prediction heads are trained jointly:

1. **Formation Energy per Atom** (regression)
2. **Energy Above Hull** (regression)
3. **Thermodynamic Stability** (binary classification)

The total loss is defined as:

```
L = MSE(FE) + MSE(Hull) + BCE(Stability)
```

This formulation enforces **thermodynamic consistency** across tasks.


### Step 4: Ensemble Training (Core Addition)

Instead of training a single model, Method 6 trains **N independent models**:

* Identical architecture
* Different random initializations
* Trained independently on the same data split

Each ensemble member produces its own predictions.


### Step 5: Ensemble Aggregation

For each material:

* **Mean prediction** → final output
* **Standard deviation across ensemble members** → epistemic uncertainty

This provides:

* `fe_mean`, `fe_std`
* `hull_mean`, `hull_std`
* `stab_mean`, `stab_std`


### Step 6: Risk-Controlled Screening

Predictions are ranked by uncertainty, enabling:

* Rejection of high-uncertainty samples
* Accuracy–coverage trade-off analysis

This transforms the model from a predictor into a **screening tool**.


## 4. Training Configuration

### Common Settings

* Optimizer: Adam
* Loss: Multi-task loss (regression + classification)
* Input data split: Train / Validation / Test
* No structural information used


### **Experiment 1: Low-Budget Ensemble**

* Ensemble size: **5**
* Epochs: **10**
* Purpose: Validate uncertainty behavior under limited training


### **Experiment 2: Fully Trained Ensemble**

* Ensemble size: **10**
* Epochs: **~100**
* Purpose: Evaluate performance under convergence


## 5. Validation Protocol

### Single-Model Validation

For each ensemble member:

* Validation MAE tracked across epochs
* Best epoch selected based on **minimum validation formation-energy MAE**

This confirms:

* Stable convergence
* Diversity across ensemble members


### Ensemble Validation

Final metrics computed using ensemble-mean predictions on the test set:

* MAE for formation energy
* MAE for hull distance
* Accuracy, F1-score, and ROC-AUC for stability


## 6. Results

### 🔹 Experiment 1: Small Ensemble, Low Epochs

| Metric                        | Result        |
| ----------------------------- | ------------- |
| FE MAE (ensemble mean)        | **0.1366 eV** |
| Hull MAE                      | **0.1081 eV** |
| Stability F1                  | **0.8839**    |
| ROC-AUC                       | **0.8750**    |
| Uncertainty–Error Correlation | **0.453**     |
| FE MAE @ 10% coverage         | **0.0804 eV** |

**Key finding:**
Even with minimal training, uncertainty correlates positively with prediction error, enabling effective early rejection.


### 🔹 Experiment 2: Large Ensemble, Full Training

| Metric                        | Result        |
| ----------------------------- | ------------- |
| FE MAE (ensemble mean)        | **0.0975 eV** |
| Hull MAE                      | **0.0749 eV** |
| Stability Accuracy            | **0.8982**    |
| Stability F1                  | **0.9306**    |
| ROC-AUC                       | **0.9557**    |
| Uncertainty–Error Correlation | **0.6396**    |
| FE MAE @ 10% coverage         | **0.0446 eV** |

**Key finding:**
Uncertainty signals strengthen with training, and risk-controlled screening achieves more than **2× error reduction** on confident predictions.


## 7. Why Method 6 Is Novel (Clear and Defensible)

Method 6 introduces **three capabilities absent from Methods 1–5**:

1. **Explicit epistemic uncertainty estimation**
2. **Risk-controlled prediction via coverage–accuracy analysis**
3. **Decision-oriented evaluation beyond single MAE values**

Importantly, this novelty is achieved **without changing the underlying architecture**, demonstrating that **reliability gains can be obtained through learning strategy rather than architectural complexity**.


## 8. Summary

> *“Method 6 extends the baseline composition-only multi-task framework into a decision-aware screening model. The strong correlation between uncertainty and error, together with systematic accuracy gains under selective prediction, provides compelling evidence that the proposed approach is suitable for early-stage materials discovery.”*

