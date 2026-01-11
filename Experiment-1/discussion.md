# Discussion

This section discusses the performance of the implemented machine learning models for space group prediction and compares the obtained results with those reported in prior literature. The discussion focuses on accuracy trends, class imbalance effects, and differences between the present study and reference works.

---

## Model-wise Performance Analysis

### XGBoost

Among the evaluated models, XGBoost achieved the highest overall performance, with an accuracy of approximately 0.49 and a macro F1-score of 0.34. The Top-k accuracy analysis shows that the model correctly identifies the true space group within the top five predictions in approximately 77% of cases and within the top ten predictions in approximately 85% of cases. This indicates that XGBoost is able to capture coarse symmetry-related patterns in the data, even when the exact space group prediction is incorrect.

However, the classification report reveals substantial variability in performance across space groups. Frequently occurring space groups exhibit relatively high precision and recall, while rare space groups show near-zero recall and F1-scores. This behavior reflects the strong class imbalance inherent in space group datasets and suggests that XGBoost prioritizes dominant classes during optimization, despite the use of multi-class learning.

---

### Random Forest

The Random Forest model achieved a Top-1 accuracy of approximately 0.42 and a macro F1-score of 0.38. While its overall accuracy is lower than that of XGBoost, the macro F1-score is higher, indicating improved performance on minority classes. This suggests that Random Forest provides better class-level balance compared to XGBoost, at the cost of reduced overall accuracy.

The Top-3 and Top-5 accuracies (approximately 0.63 and 0.72, respectively) demonstrate that the Random Forest model can often narrow down the correct space group to a small candidate set. The classification report further shows that recall values for many minority space groups are higher than those obtained with XGBoost, albeit often accompanied by low precision. This pattern is characteristic of ensemble tree-based methods when class weights are applied to mitigate imbalance.

---

### Multilayer Perceptron (MLP)

The MLP model exhibited the weakest performance among the evaluated approaches, with an accuracy of approximately 0.39 and a macro F1-score of 0.23. Although the Top-5 and Top-10 accuracies exceed 0.67 and 0.78, respectively, the per-class metrics indicate that the model struggles to learn discriminative representations for a large number of space groups.

The classification report shows consistently low recall for rare classes and limited gains for frequent classes, suggesting that the fully connected neural network is insufficient for capturing the complex, symmetry-related relationships present in composition-based descriptors without specialized feature engineering or architectural modifications.

---

## Comparison with Prior Work

The obtained results are notably lower than those reported in prior studies on space group prediction. In the ACS Omega (2020) study, Random Forest models trained on carefully engineered Magpie descriptors achieved F1-scores of approximately 0.65 for multiclass space group prediction, albeit on a restricted set of the most frequent space groups. Similarly, the Journal of Applied Crystallography (2024) study reported a Top-1 accuracy of approximately 0.56 and a Top-5 accuracy of approximately 0.87 when predicting 172 space groups using extended composition-based descriptors and Random Forest models.

In contrast, the present study evaluates a substantially larger and more imbalanced set of space groups, including many classes with very limited sample sizes. This difference in experimental setup significantly impacts achievable performance and partially explains the observed performance gap. The lower Top-1 accuracy and macro F1-scores indicate that predicting fine-grained space group labels from composition alone remains challenging, particularly when no explicit structural or symmetry-aware descriptors are incorporated.

---

## Impact of Class Imbalance and Descriptor Limitations

Across all models, the classification reports consistently show poor recall for space groups with small sample sizes. This highlights class imbalance as a dominant limiting factor. While ensemble methods such as Random Forest improve recall for minority classes relative to boosting and neural network approaches, none of the evaluated models fully overcome this issue.

Furthermore, the descriptors used in this study are purely composition-based and do not explicitly encode crystallographic symmetry operations or lattice information. As a result, models are often able to predict symmetry families or closely related space groups, as reflected in the relatively strong Top-k accuracy, but struggle to uniquely identify the exact space group.

---

## Overall Observations

Overall, the results demonstrate that composition-based machine learning models can provide meaningful probabilistic rankings of candidate space groups but exhibit limited see-saw trade-offs between accuracy and class-level fairness. XGBoost offers higher overall accuracy, Random Forest provides better macro-level balance, and MLP underperforms in both respects. Compared to prior studies, the reduced performance underscores the sensitivity of space group prediction to dataset composition, feature engineering, and class selection criteria.

These findings reinforce the conclusion that space group prediction from composition alone is inherently difficult and that careful feature construction, class filtering, and evaluation design are critical for achieving performance comparable to state-of-the-art literature.
