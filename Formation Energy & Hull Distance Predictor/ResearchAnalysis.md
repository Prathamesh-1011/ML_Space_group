# 🔬 Comparative Analysis: Formation Energy & Hull Distance Prediction Models


# 1️⃣ **CrabNet**

## 1.1 Objective (Stated Explicitly)

* Predict **materials properties using composition only**
* Formation energy is the **primary benchmark task**

📍 *Page 2, Introduction*

## 1.2 Dataset

### Source

* **Materials Project (MP)**
* ~**132,000 inorganic compounds**

📍 *Page 3, “Datasets” section*

### Target Variable

* **Formation energy per atom (eV/atom)**
* DFT-computed, PBE functional


## 1.3 Data Preparation & Preprocessing

### Composition Parsing

* Chemical formula parsed into:

  * List of elements
  * Corresponding stoichiometric fractions

📍 *Page 4, Figure 1 caption*

### Encoding

* **No handcrafted features**
* Each element represented by:

  * Learned embedding vector
  * Explicit **fractional amount**

> ❗ No Magpie, no matminer, no physical descriptors

📍 *Page 4*

### Normalization

* Target normalized to zero mean, unit variance during training

📍 *Page 5, Training details*


## 1.4 Model Architecture

### Core Model

* **Transformer-based self-attention network**

📍 *Page 4, Figure 1*

### Architecture Components

1. Element embedding layer
2. Fractional embedding concatenation
3. Multi-head self-attention blocks
4. Feed-forward projection
5. Mean pooling
6. Regression head

📍 *Page 4–5*

### Key Innovation

* **Fractional encoding** → attention weighted by stoichiometry


## 1.5 Training Methodology

* Optimizer: **Adam**
* Loss: **Mean Absolute Error (MAE)**
* Learning rate scheduling: yes
* Batch size: specified in appendix

📍 *Page 5*


## 1.6 Validation Strategy

* Random **train/validation/test split**
* No compositional holdout
* No leave-one-element-out validation

📍 *Page 6*


## 1.7 Results (Exact Location)

### Formation Energy (MP)

* **MAE = 0.0296 eV/atom**

📍 **Page 6, Table 1**


## 1.8 Limitations (Explicitly Stated)

* No uncertainty quantification
* No structural sensitivity
* Limited extrapolation to unseen chemistries

📍 *Page 9, Discussion*


# 2️⃣ **Roost**

## 2.1 Objective

* Predict formation energy **without crystal structure**
* Learn chemically meaningful element embeddings

📍 *Page 2*

## 2.2 Dataset

### Source

* **OQMD** (~256k compounds)
* **Materials Project** used for transfer

📍 *Page 3*

### Target

* Formation energy per atom


## 2.3 Data Preparation

### Composition Graph Construction

* Nodes = elements
* Fully connected graph
* Edge weights = product of element fractions

📍 *Page 3, Figure 1*

### Element Initialization

* Pretrained **MatScholar embeddings**

📍 *Page 3*


## 2.4 Model Architecture

### Model Type

* **Message Passing Neural Network (MPNN)**

### Architecture

1. Element embedding
2. Fraction-weighted message passing
3. Attention aggregation
4. Global pooling
5. Regression head

📍 *Page 3–4*


## 2.5 Training Methodology

* Optimizer: Adam
* Loss: MAE
* Ensemble of **10 independent models**

📍 *Page 4*


## 2.6 Validation

* Hold-out test set
* Learning curves vs dataset size
* Uncertainty calibration curves

📍 *Page 4–5*


## 2.7 Results

### Formation Energy

* Single model: **MAE = 0.0297 eV**
* Ensemble: **MAE = 0.0241 eV**

📍 **Page 4, Table 1**


## 2.8 Limitations

* No structure → polymorphs indistinguishable
* Higher inference cost than CrabNet

📍 *Page 6*


# 3️⃣ **CGCNN**


## 3.1 Objective

* Predict formation energy **from crystal structure**

📍 *Page 1*


## 3.2 Dataset

* Materials Project
* ~47,000 relaxed structures

📍 *Page 4*


## 3.3 Data Processing

### Graph Construction

* Nodes = atoms
* Edges = neighbors within cutoff
* Edge features = Gaussian-expanded distances

📍 *Page 2*


## 3.4 Architecture

* Crystal graph convolution layers
* Pooling → dense regression head

📍 *Page 2–3*


## 3.5 Validation

* Random split
* MAE metric

📍 *Page 5*


## 3.6 Results

* Formation Energy MAE = **0.039 eV/atom**

📍 **Page 5, Table 2**


## 3.7 Limitations

* Requires structure
* Lower accuracy than newer GNNs

📍 *Page 6*


# 4️⃣ **MEGNet**


## 4.1 Objective

* Universal graph network for molecules & crystals

📍 *Page 1*


## 4.2 Dataset

* Materials Project (~69k)

📍 *Page 2*


## 4.3 Data Processing

* Nodes: atoms
* Edges: bonds
* Global state vector (optional)

📍 *Page 2*


## 4.4 Architecture

* Graph network with:

  * Node update
  * Edge update
  * State update
* Set2Set pooling

📍 *Page 2–3*


## 4.5 Validation

* Transfer learning evaluation
* Cross-property tests

📍 *Page 4*


## 4.6 Results

* Formation Energy MAE = **0.028 eV/atom**

📍 **Page 3, Table 1**


## 4.7 Limitations

* Needs structure
* Computationally heavier

📍 *Page 5*


# 5️⃣ **DeeperGATGNN**

## 5.1 Objective

* Overcome GNN oversmoothing
* Improve formation energy prediction

📍 *Page 1*

## 5.2 Dataset

* MatBench datasets
* MP-derived formation energy sets

📍 *Page 7*


## 5.3 Architecture

* Graph Attention Network
* 20–30 layers
* Residual connections + group normalization

📍 *Page 6*


## 5.4 Validation

* MatBench benchmark protocol

📍 *Page 8*


## 5.5 Results

* Formation Energy MAE ≈ **0.032 eV/atom**

📍 **Page 8, Table 3**


## 5.6 Limitations

* Requires large data
* Not suitable for early screening

📍 *Page 10*


# 🎯 FINAL COMPARATIVE CONCLUSION (FACT-BASED)

| Model            | Structure Needed | FE MAE (eV) | Page |
| ---------------- | ---------------- | ----------- | ---- |
| CrabNet          | ❌                | **0.0296**  | p6   |
| Roost (ensemble) | ❌                | **0.0241**  | p4   |
| CGCNN            | ✅                | 0.039       | p5   |
| MEGNet           | ✅                | 0.028       | p3   |
| DeeperGATGNN     | ✅                | 0.032       | p8   |
