# 🔬 Comparative Analysis: Formation Energy & Hull Distance Prediction Models

---

## 1️⃣ **Composition-Only Models (Stage-1 Screening, No Structure)**

These are **closest to your Model-2 design philosophy**.

---

## **CrabNet**

### Model Type

* Transformer with **fractional stoichiometry encoding**
* Multi-head self-attention over elements

### Inputs

* Chemical formula only (elements + fractions)

### Dataset

* Materials Project (≈132k compounds)

### Data Processing

* Formula parsing → element tokens
* Fractional encoding (explicit stoichiometry)
* No handcrafted descriptors

### Methodology

* Attention learns **inter-element interactions**
* End-to-end regression
* Supports **multi-task learning** (shared backbone)

### Targets

* Formation Energy per atom
* (Hull distance inferred downstream or via auxiliary task)

### Results

* **Formation Energy MAE ≈ 0.0296 eV/atom**
* Comparable to structure-based GNNs

### Validation

* Random train/val/test split
* MAE, RMSE
* Cross-dataset benchmarks (MatBench)

### Strengths

* Best **composition-only accuracy**
* Interpretability (attention maps)
* Fast inference → ideal pre-DFT filter

### Limitations

* No explicit uncertainty unless ensembled
* Hull distance not primary target (but derivable)

---

## **Roost**

### Model Type

* Message-Passing Neural Network (MPNN)
* Dense **composition graph**

### Inputs

* Chemical formula (nodes = elements, weighted by fraction)

### Dataset

* OQMD (≈256k)
* Materials Project (≈132k)

### Data Processing

* Element embeddings (MatScholar)
* Fraction-weighted graph construction

### Methodology

* Attention-based message passing
* Deep ensembles for uncertainty

### Targets

* Formation Energy per atom
* (Hull distance via post-processing)

### Results

* **Single model MAE ≈ 0.0297 eV/atom**
* **Ensemble MAE ≈ 0.0241 eV/atom**

### Validation

* Hold-out test set (10%)
* Learning curves (sample efficiency)
* Calibration via confidence–error curves

### Strengths

* **Uncertainty quantification**
* Very data-efficient
* Excellent for screening unknown chemistries

### Limitations

* Less interpretable than CrabNet
* Slightly slower inference

---

## 2️⃣ **Structure-Based GNNs (Stage-2 / Stage-3, Require Structure)**

These are **not composition-only**, but important benchmarks.

---

## **CGCNN**

### Model Type

* Graph Convolutional Neural Network

### Inputs

* Full crystal structure (atomic positions + neighbors)

### Dataset

* Materials Project (~47k)

### Data Processing

* Graph construction from relaxed structures
* Distance-based Gaussian edge features

### Methodology

* Convolution → pooling → regression

### Targets

* Formation Energy per atom

### Results

* **MAE ≈ 0.039 eV/atom**

### Validation

* Random splits
* MAE, RMSE across properties

### Strengths

* First crystal-GNN baseline
* Handles polymorphs

### Limitations

* Requires structure
* Lower accuracy than modern GNNs

---

## **MEGNet**

### Model Type

* Graph Network with **global state vector**

### Inputs

* Crystal structure + optional state variables

### Dataset

* Materials Project (~69k)

### Data Processing

* Graph with nodes, edges, global state
* Transfer-learning capable

### Methodology

* Message passing with node–edge–state updates

### Targets

* Formation Energy
* Energy Above Hull (via derived stability)

### Results

* **Formation Energy MAE ≈ 0.028 eV/atom**

### Validation

* Cross-property transfer learning
* MAE on multiple tasks

### Strengths

* Better accuracy than CGCNN
* Handles thermodynamic states

### Limitations

* Needs structure
* Slower and heavier than composition models

---

## **DeeperGATGNN**

### Model Type

* Deep Graph Attention Network (20–30 layers)

### Inputs

* Crystal structure

### Dataset

* Large MP-derived benchmarks (>50k)

### Data Processing

* Deep residual graph construction
* Group normalization

### Methodology

* Very deep message passing
* Skip connections to avoid oversmoothing

### Targets

* Formation Energy

### Results

* **MAE ≈ 0.032 eV/atom**
* Better than shallow GNNs at scale

### Validation

* MatBench benchmarks
* Large-data regime evaluation

### Strengths

* Scales well with big data
* High representational power

### Limitations

* Heavy compute
* Not suitable for early screening

---

## 3️⃣ **ML Potentials / Stability-Focused Models**

---

## **MACE**

*(from Matbench Discovery)*

### Model Type

* Universal Interatomic Potential (forces + energies)

### Inputs

* Atomic structure (unrelaxed or relaxed)

### Dataset

* Materials Project (≈154k)
* WBM hypothetical set (≈257k)

### Targets

* Formation Energy
* Energy Above Hull
* Stability classification

### Results

* **F1 ≈ 0.78** (stability)
* Discovers **~62% stable materials at 10% FPR**

### Validation

* Matbench Discovery benchmark
* Discovery Acceleration Factor (DAF)

### Strengths

* Best **hull-distance/stability performance**
* DFT-like accuracy

### Limitations

* Requires structure
* Overkill for Stage-1 screening

---

# 📊 Summary Comparison Table (Only Relevant Models)

| Model                | Input       | Formation Energy MAE (eV/atom) | Hull Distance | Uncertainty | Best Use Stage |
| -------------------- | ----------- | ------------------------------ | ------------- | ----------- | -------------- |
| **CrabNet**          | Composition | **0.0296**                     | Indirect      | ❌           | Stage-1        |
| **Roost (Ensemble)** | Composition | **0.0241**                     | Indirect      | ✅           | Stage-1        |
| CGCNN                | Structure   | 0.039                          | ❌             | ❌           | Stage-2        |
| MEGNet               | Structure   | 0.028                          | ✅             | ❌           | Stage-2        |
| DeeperGATGNN         | Structure   | 0.032                          | ❌             | ❌           | Stage-2/3      |
| MACE                 | Structure   | ~DFT-level                     | **✅**         | ✅           | Stage-3        |

---

# 🎯 Final Takeaway for *Your Model-2*

Your **Formation Energy & Hull Distance Predictor** is **most directly aligned with**:

> **CrabNet + Roost (Multi-Task / Ensemble)**

because:

* Same **input assumption** (composition-only)
* Same **dataset scale** (MP ~132k)
* Same **goal** (pre-DFT stability filtering)
* Easily extensible to **multi-task learning**
* Supported by **Matbench Discovery philosophy**

