# 1. Dataset Creation and CSV Files

### 1.1 Data Source

The dataset was constructed using the **Materials Project (MP) database** via the `mp-api`. Only **thermodynamically stable and near-stable compounds** were considered by filtering materials with:

* **Energy above hull:** `0 ≤ E_hull ≤ 0.1 eV`

This ensures the dataset contains physically meaningful crystal structures.

---

### 1.2 Raw Dataset (`mp_spacegroup_data.csv`)

Each material entry includes:

* `formula` – Reduced chemical formula
* `spacegroup` – International space group number (1–230)
* `energy_above_hull` – Stability indicator

This file represents the **raw crystallographic classification dataset** before machine learning preprocessing.

---

### 1.3 Train / Validation / Test Split

To enable unbiased model training and evaluation:

* **Train:** 80%
* **Validation:** 10%
* **Test:** 10%

Stratification was applied during the **train–temp split** to preserve space-group distribution. Classes with fewer than two samples were removed to avoid degenerate learning cases.

Generated files:

* `train_data.csv`
* `val_data.csv`
* `test_data.csv`

Each file contains:

* `formula`
* `spacegroup` (encoded label)

---

### 1.4 Feature Engineering (`*_features.csv`)

Chemical formulas were converted into **numerical descriptors** using **MatMiner**:

#### Featurizers Used

* **MAGPIE elemental statistics**
* **Stoichiometric descriptors**

Total descriptors: **138 composition-based features**

Final ML-ready datasets:

* `train_features.csv`
* `val_features.csv`
* `test_features.csv`

Each row represents:

* 138 numerical descriptors (X)
* `spacegroup` class label (y)

---

# 2. Data Profiling and Exploratory Data Analysis (EDA)

### 2.1 Target Variable Analysis (y)

* The dataset exhibits **strong class imbalance** across space groups
* A small subset of space groups dominates the distribution
* Many space groups appear with very few samples

This imbalance motivated:

* Use of **macro-F1 score**
* **Top-k accuracy metrics**
* Careful handling of oversampling methods

---

### 2.2 Feature Distribution Analysis (X)

* Several descriptors show **right-skewed and heavy-tailed distributions**
* Some features are near-constant for many materials
* Significant scale differences exist across descriptors

This justified:

* Feature standardization
* Correlation-based inspection
* Dimensionality-aware models

---

### 2.3 Feature Correlation Analysis

* Strong correlations exist among elemental statistics (mean, max, range)
* Redundancy observed within MAGPIE feature families
* No single descriptor dominates prediction alone

Correlation maps and heatmaps were used to:

* Identify multicollinearity
* Understand chemically meaningful feature groupings

---

# 3. Machine Learning Models Trained

All models were trained on `train_features.csv` and evaluated on `test_features.csv`.

---

## 3.1 XGBoost (Gradient Boosted Trees)

### Configuration

* Objective: `multi:softprob`
* Features standardized
* Class imbalance handled implicitly
* Hyperparameter tuning via cross-validation

### Results

* **Accuracy:** 0.4745
* **Macro F1:** 0.3311

**Top-k Accuracy**

* Top-1: 0.4745
* Top-5: 0.7669
* Top-10: 0.8545

### Interpretation

XGBoost performed best overall, demonstrating strong capability in capturing **nonlinear compositional–symmetry relationships**.

---

## 3.2 Random Forest

### Attempted Configuration

* High-dimensional input (138 features)
* Large number of classes (>200)

### Outcome

* Model exhausted available RAM
* Kernel restart occurred

### Conclusion

Standard Random Forests are **memory-inefficient** for:

* Large multi-class problems
* High-dimensional feature spaces

Tree ensembles without boosting are unsuitable under limited memory constraints.

---

## 3.3 Multi-Layer Perceptron (MLP)

### Configuration

* Architecture: `(128, 64)`
* Standardized inputs
* Iterative optimization

### Results

* **Accuracy:** 0.3864
* **Macro F1:** 0.2269

**Top-k Accuracy**

* Top-1: 0.3864
* Top-5: 0.6773
* Top-10: 0.7822

### Interpretation

MLP captured global feature interactions but struggled with:

* Severe class imbalance
* Fine-grained crystallographic distinctions

---

# 4. Key Observations and Learnings

* Composition-only features can predict space groups with **moderate accuracy**
* Tree boosting outperforms deep fully-connected networks in this regime
* Class imbalance is the dominant limiting factor
* Random Forests are impractical at this scale without heavy optimization
* Top-k metrics are essential for crystallography prediction tasks

---

# 5. Current State of the Project

* ✔ Dataset construction
* ✔ Feature engineering (138 descriptors)
* ✔ Full data profiling
* ✔ XGBoost and MLP evaluation
* ✔ Identification of RF limitations

---
