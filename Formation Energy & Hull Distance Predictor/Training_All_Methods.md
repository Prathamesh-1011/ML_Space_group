# Training Details for All Methods

This document summarizes **training configurations, best model architectures, validation strategy, and final results** for each method and experiment exactly as implemented in the submitted notebooks. The goal is to clearly document *how each model was trained* and *what performance it achieved*, in a reproducible manner.

---

## Common Dataset and Evaluation Setup

* **Dataset**: Materials Project–derived composition dataset
* **Total samples**: 150,200
* **Input**: Chemical formula only
* **Targets**:

  * Formation energy per atom (regression)
  * Energy above hull (regression)
  * Stability (binary classification)
* **Split**:

  * Training: 80% (120,160 samples)
  * Validation: 10% (15,020 samples)
  * Test: 10% (15,020 samples)
* **Evaluation metrics**:

  * MAE for formation energy
  * MAE for hull distance
  * F1-score for stability

---

## Method 1: Transformer-Based Composition Multi-Task Network (TC-MTL)

### Best Model Architecture

* Element embedding dimension: 256
* Transformer encoder:

  * Layers: 6
  * Attention heads: 8
* Fractional encoding applied to embeddings
* Pooling: Sum over elements
* Shared dense layers: 512 → 256
* Output heads:

  * Formation energy (regression)
  * Hull distance (regression)
  * Stability (sigmoid classification)

### Training Parameters

* Optimizer: AdamW
* Learning rate: 3e-4
* Weight decay: 1e-4
* Batch size: 256
* Epochs: up to 10
* Scheduler: ReduceLROnPlateau

  * Factor: 0.5
  * Patience: 4
* Early stopping patience: 8

### Loss Function

* 1.5 × SmoothL1Loss (formation energy)
* 0.5 × L1Loss (hull distance)
* 0.1 × BCELoss (stability)

### Validation Strategy

* Validation monitored on **formation energy loss only**
* Best model checkpoint saved based on lowest validation loss

### Test Results

* Formation Energy MAE: 0.139 eV/atom
* Hull Distance MAE: 0.076 eV
* Stability F1-score: 0.898

---

## Experiment: Increased Training Duration (TC-MTL)

### Change from Method 1

* Same architecture and loss configuration
* Training continued until validation plateau

### Observation

* Training loss decreased with more epochs
* Validation loss saturated early
* No significant improvement beyond early convergence

### Conclusion

Increasing epochs alone did not significantly improve generalization performance.

---

## Method 2: Lightweight Transformer Composition Multi-Task Network (LTC-MTL)

### Best Model Architecture

* Element embedding dimension: 128
* Transformer encoder:

  * Layers: 3
  * Attention heads: 4
* Pooling: Sum over elements
* Shared dense layers: 256 → 256
* Three task-specific output heads

### Training Parameters

* Optimizer: AdamW
* Learning rate: 1e-3
* Batch size: 256
* Epochs: 50
* Scheduler: None
* Early stopping: None

### Loss Function

* 1.0 × L1Loss (formation energy)
* 0.7 × L1Loss (hull distance)
* 0.3 × BCELoss (stability)

### Validation Strategy

* Training monitored via training loss only
* No early stopping or scheduler

### Test Results

* Formation Energy MAE: 0.120 eV/atom
* Hull Distance MAE: 0.071 eV
* Stability F1-score: 0.896

---


## Method 3: Ensemble of Transformer Composition Multi-Task Networks (E-TC-MTL)

### Model Architecture

* Same as Method 2 (LTC-MTL)

### Training Parameters

* Number of ensemble members: 10
* Epochs per model: 5
* Optimizer: AdamW
* Learning rate: 1e-3
* Batch size: 256

### Inference Strategy

* Regression outputs averaged across models
* Stability predicted using mean probability thresholded at 0.5

### Test Results (Ensemble Mean)

* Formation Energy MAE: 0.129 eV/atom
* Hull Distance MAE: 0.082 eV
* Stability F1-score: 0.888

---

## Method 4: Optimized Training with Scheduler and Early Stopping (O-LTC-MTL)

### Change from Method 2

* Architecture unchanged
* Added learning rate scheduler and early stopping

### Training Parameters

* Optimizer: AdamW
* Initial learning rate: 1e-3
* Scheduler: ReduceLROnPlateau

  * Factor: 0.5
  * Patience: 3
* Early stopping patience: 5
* Maximum epochs: 50 (early stopped)

### Validation Strategy

* Validation monitored using formation energy loss
* Best model saved automatically

### Test Results (Best Single Model)

* Formation Energy MAE: 0.104 eV/atom
* Hull Distance MAE: 0.066 eV
* Stability F1-score: 0.909

---

## Method 5: Feature-Based Multi-Task Learning with Curriculum Training (FB-MTL)

### Input Representation

* MAGPIE composition descriptors (~145 features)
* StandardScaler normalization

### Best Model Architecture

* Shared MLP: 512 → 256
* Batch normalization applied
* Output heads:

  * Formation energy (regression)
  * Hull distance (regression)
  * Stability (classification using latent + hull)

### Training Parameters

* Optimizer: AdamW
* Learning rate: 1e-3
* Scheduler: CosineAnnealingLR (T_max = 50)
* Epochs: 50
* Batch size: 256

### Special Training Strategies

* Near-hull oversampling during training
* Curriculum learning:

  * Epochs 0–15: formation energy only
  * Epochs 15–30: formation energy + hull distance
  * Epochs 30–50: all three tasks
* Class-weighted BCEWithLogitsLoss for stability

### Validation Strategy

* Validation MAE tracked for formation energy and hull distance
* Stability threshold optimized on validation set

### Final Test Results

* Formation Energy MAE: 0.103 eV/atom
* Hull Distance MAE: 0.059 eV
* Stability F1-score: 0.819

---

## Summary of Best Results Across Methods

| Method                 | FE MAE (↓) | Hull MAE (↓) | Stability F1 (↑) |
| ---------------------- | ---------- | ------------ | ---------------- |
| TC-MTL (base)          | 0.139      | 0.076        | 0.898            |
| LTC-MTL                | 0.120      | 0.071        | 0.896            |
| Ensemble (E-TC-MTL)    | 0.129      | 0.082        | 0.888            |
| O-LTC-MTL + optimization | 0.104      | 0.066        | **0.909**        |
| Feature-based FB-MTL   | **0.103**  | **0.059**    | 0.819            |

---
