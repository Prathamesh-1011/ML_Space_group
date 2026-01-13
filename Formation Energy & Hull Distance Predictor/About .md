# Model 2: Formation Energy & Hull Distance Predictor

## Purpose

The primary purpose of **Model 2** is to pre-screen which compositions are likely stable by predicting formation energy and hull distance. This model filters out potentially unstable materials before expensive Density Functional Theory (DFT) calculations, significantly reducing computational costs.

---

## Architecture

The model utilizes **Multi-task Learning (MTL)** to simultaneously predict multiple related properties.

### Why Multi-task Learning?

* **Shared representations improve generalization:** The model learns shared features that benefit all tasks.
* **Physical Relationships:** Formation energy helps predict hull distance; the model leverages the intrinsic physical correlation between these properties.
* **Performance Boost:** Multi-task learning typically outperforms single-task models by providing a more holistic "understanding" of the material space.

---

## Data Requirements

### Input Features

The model requires **composition-only inputs**, meaning no structural data (crystal lattice) is needed:

* **Chemical formula:** (e.g., `Fe2O3`, `Li3FeO4`)
* **Stoichiometry information**
* **Element properties:** Fractional contributions and intrinsic elemental traits.

### Target Variables

| Variable | Type | Description |
| --- | --- | --- |
| **Formation Energy per atom** | Regression | Energy relative to reference elements in their standard states. |
| **Energy above hull ()** | Regression | Distance from the convex hull of stability. |
| **Stability Classification** | Binary | Whether  eV/atom (potential synthesizability). |

### Training Datasets

* **Materials Project:** ~132,000 materials.

---

## Previous Research Papers & Methods

### Key Supporting Models

1. **CrabNet (Compositionally Restricted Attention-Based Network)**
* **Architecture:** Transformer-based attention mechanism.
* **Innovation:** Fractional encoding that explicitly includes stoichiometry.
* **Pros:** High interpretability through attention weights.


2. **Roost (Recurrent Orbit-based Optimization for Structure-property relationships)**
* **Architecture:** Message-passing neural network on composition graphs.
* **Innovation:** Learned element embeddings capturing periodic trends.
* **Pros:** Fast inference and built-in uncertainty quantification.


3. **MEGNet (Materials Graph Networks)**
* **Note:** Only usable if structure data is available; otherwise, CrabNet/Roost are preferred for early-stage screening.



---

## Performance Benchmarks

### Formation Energy Prediction

| Model | Architecture | MAE (eV/atom) | Training Time |
| --- | --- | --- | --- |
| **CrabNet** | Transformer | 0.0296 | 2–6 hours |
| **Roost** | Message-passing | 0.0306 | 4 hours |
| **MEGNet** | Graph NN | 0.0280 | 8 hours |
| **CGCNN** | Graph Conv | 0.0390 | 4 hours |
| **Random Forest** | Traditional ML | 0.0470 | 1 hour |

### Hull Distance Prediction

* **Expected MAE:** ~0.05–0.08 eV/atom.
* **Stability Classification:** Expected F1-score of 0.40–0.50 for composition-only inputs.

---

## Recommended Implementation Approach

1. **Primary Choice:** Start with **CrabNet** for interpretability. Alternatively, use **Roost** if fast inference and uncertainty quantification are priorities.
2. **Ensembling:** Use an ensemble of both models for maximum robustness.
3. **Handling Functional Differences:** Since different databases use different DFT functionals (PBE, OptB88-vdW), a "Delta-Learning" or functional-specific embedding approach should be implemented to normalize energy values.

---
