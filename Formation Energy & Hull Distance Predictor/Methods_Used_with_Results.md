# Composition-Only Prediction of Thermodynamic Stability

## Overview

This contains our research on predicting key thermodynamic properties of inorganic materials **using only their chemical composition**.

The core question I address is:

> Given only a chemical formula (e.g., BaTiO₃), can we efficiently estimate
>
> 1. Material stability
> 2. Distance from the thermodynamic convex hull
> 3. Formation energy per atom

This is done **without crystal structures**, **without DFT calculations**, and **without graph-based representations**.
All predictions rely purely on elemental composition.

I explore **five different learning strategies** to solve the same problem.
The task remains identical across all methods; only the *learning mechanism* changes.

---

## Dataset

I use a dataset of **150,201 inorganic materials**, where each entry contains:

* `formula` – chemical composition (e.g., LiFePO₄)
* `formation_energy_per_atom` (DFT-derived)
* `energy_above_hull` (DFT-derived)
* `is_stable` (binary stability label)

### Cleaning and Preparation

* Rows with missing formulas or target values are removed
* The dataset is shuffled to avoid ordering bias
* The final usable dataset contains **150,200 materials**

This dataset provides high-quality DFT ground truth, and the goal of all models is to learn a fast and inexpensive approximation to these properties.

---

## Data Splitting

I split the dataset as follows:

* 80% for training
* 10% for validation
* 10% for testing

This ensures proper generalization, avoids memorization, and enables fair performance reporting.

---

## Composition Encoding Pipeline (Common to All Methods)

### Formula Parsing

Each chemical formula is decomposed into:

* A list of constituent elements
* Corresponding stoichiometric fractions

Example:

```
Ba₂TiO₄ → Elements: [Ba, Ti, O]
           Fractions: [2/7, 1/7, 4/7]
```

This representation encodes composition without introducing any structural information.

### Element Indexing

* The dataset contains **89 unique elements**
* Each element is mapped to a unique integer index
* This allows models to learn element-specific representations automatically

### Padding

Since different materials contain different numbers of elements, shorter compositions are padded with zeros so that batches can be processed efficiently.

---

## Method 1: Transformer-Based Composition Multi-Task Network (TC-MTL)

### Purpose

This is my most advanced single-model architecture, designed to maximize predictive accuracy while maintaining training stability.

### Architecture

* Learnable element embeddings (256 dimensions)
* Fractional encoding to incorporate stoichiometry
* Transformer encoder to model interactions between elements
* Pooling to obtain a single composition-level representation
* Shared neural network followed by three task-specific heads:

  * Formation energy regression
  * Energy above hull regression
  * Stability classification

### Training Strategy

* Formation energy and hull distance targets are scaled
* Loss weighting prioritizes formation energy
* Learning rate scheduling is applied
* Early stopping prevents overfitting

### Test Performance

| Property             | Value         |
| -------------------- | ------------- |
| Formation Energy MAE | 0.139 eV/atom |
| Hull Distance MAE    | 0.076 eV      |
| Stability F1 Score   | 0.898         |

---

## Method 2: Lightweight Transformer Composition Multi-Task Network (LTC-MTL)

### Purpose

To test whether architectural complexity is necessary for strong performance.

### Key Differences from Method 1

* Smaller embedding size (128)
* Fewer transformer layers (3)
* No target scaling
* No learning rate scheduler

### Test Performance

| Property             | Value         |
| -------------------- | ------------- |
| Formation Energy MAE | 0.120 eV/atom |
| Hull Distance MAE    | 0.071 eV      |
| Stability F1 Score   | 0.896         |

### Insight

Despite being significantly simpler, this model achieves nearly the same performance as the larger transformer, indicating that **problem formulation is more important than architectural depth**.

---

## Method 3: Ensemble of Transformer-Based Models (E-TC-MTL)

### Purpose

To improve robustness and reduce prediction variance.

### Approach

* Train 10 independent transformer models
* Each model differs due to random initialization and data ordering
* Final predictions are obtained by averaging outputs

### Test Performance

| Property             | Value         |
| -------------------- | ------------- |
| Formation Energy MAE | 0.129 eV/atom |
| Hull Distance MAE    | 0.082 eV      |
| Stability F1 Score   | 0.888         |

### Insight

While not outperforming the best single model, the ensemble provides more reliable and stable predictions, making it useful for uncertainty-aware screening.

---

## Method 4: Optimized Transformer Composition Multi-Task Network (O-TC-MTL)

### Purpose

To improve generalization by optimizing training dynamics rather than changing model architecture.

### Improvements

* Adaptive learning rate reduction
* Validation-based early stopping

### Test Performance (Best Single Model)

| Property             | Value         |
| -------------------- | ------------- |
| Formation Energy MAE | 0.104 eV/atom |
| Hull Distance MAE    | 0.066 eV      |
| Stability F1 Score   | 0.909         |

### Key Result

This model achieves the **best overall performance** among all composition-only neural approaches tested.

---

## Method 5: Feature-Based Multi-Task Learning with Curriculum Training (FB-MTL)

### Purpose

To compare learned representations against hand-crafted physical descriptors.

### Input Representation

Instead of raw compositions, this method uses **MAGPIE descriptors**, including:

* Mean electronegativity
* Atomic radius variance
* Valence electron statistics

### Key Innovations

* Oversampling of near-hull materials
* Curriculum learning:

  1. Learn formation energy
  2. Learn energy above hull
  3. Learn stability classification
* Physics-aware dependency where stability prediction depends on hull prediction

### Test Performance

| Property             | Value         |
| -------------------- | ------------- |
| Formation Energy MAE | 0.103 eV/atom |
| Hull Distance MAE    | 0.059 eV      |
| Stability F1 Score   | 0.819         |

### Insight

This method achieves the best hull distance accuracy but underperforms in stability classification, while maintaining strong physical consistency.

---

## Final Comparison

| Method          | FE MAE ↓ | Hull MAE ↓ | Stability F1 ↑ |
| --------------- | -------- | ---------- | -------------- |
| TC-MTL          | 0.139    | 0.076      | 0.898          |
| LTC-MTL         | 0.120    | 0.071      | 0.896          |
| E-TC-MTL        | 0.129    | 0.082      | 0.888          |
| O-TC-MTL        | 0.104    | 0.066      | 0.909          |
| FB-MTL (MAGPIE) | 0.103    | 0.059      | 0.819          |

---

## Key Takeaways

* Composition-only models can accurately approximate thermodynamic properties
* Simple architectures combined with good training strategies outperform overly complex designs
* Multi-task learning improves physical consistency
* This work emphasizes **pipeline design and learning strategy**, not leaderboard chasing

---
