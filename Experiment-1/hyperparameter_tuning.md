
# Hyperparameter Tuning Record

**Task:** Space Group Prediction (Multiclass, 221 classes)
**Features:** MAGPIE-based tabular features (after PCA)
**Models:** XGBoost
**Environment:** AWS EC2 (CPU-based)

---

## 1. Dataset & Preprocessing Summary

* Total space groups: **221**
* Feature preprocessing:

  * Removed redundant norm features (except 2-norm)
  * Removed highly correlated features (>0.97)
  * Log transform on highly skewed features
  * StandardScaler
  * PCA (95% variance retained → **37 components**)
* Class imbalance handling:

  * SMOTE (training set only)

---

## 2. Hyperparameter Search Strategy (XGBoost)

* Optimization framework: **Optuna**
* Objective metric: **Macro F1-score**
* Validation strategy: **Held-out validation set**
* Objective function:

  ```
  maximize macro_f1(y_val, y_pred)
  ```

---

## 3. Hyperparameter Search Space

| Parameter        | Range      |
| ---------------- | ---------- |
| n_estimators     | 200 – 800  |
| max_depth        | 6 – 15     |
| learning_rate    | 0.01 – 0.2 |
| subsample        | 0.7 – 1.0  |
| colsample_bytree | 0.7 – 1.0  |
| reg_alpha (L1)   | 0.0 – 5.0  |
| reg_lambda (L2)  | 0.0 – 5.0  |

Fixed parameters:

* objective = `multi:softprob`
* num_class = 221
* n_jobs = -1
* random_state = 42

---

## 4. Best Hyperparameters (Optuna Output)

### ✅ **XGBoost – Best Configuration**

```yaml
model: XGBoost
n_estimators: 563
max_depth: 11
learning_rate: 0.0110
subsample: 0.8758
colsample_bytree: 0.9186
reg_alpha: 4.1082
reg_lambda: 1.9420
objective: multi:softprob
num_class: 221
random_state: 42
```

---

## 5. Final Model Training Configuration

### XGBoost (Final)

```yaml
training_data: PCA-transformed training set
validation_data: held-out validation set
loss_function: multiclass log-loss
evaluation_metric: macro F1
```


---

## 6. Final Evaluation Results (TEST SET)

> **Fill these after final run**

### XGBoost Results

* **Accuracy:** `53.67%`
* **Top-3 Accuracy:** `53.67`
* **Top-5 Accuracy:** `75.87`
* **Macro F1-score:** `87.64`

---




