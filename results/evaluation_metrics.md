# ===================================================
# ==   Heart Disease Prediction - Model Evaluation   ==
# ===================================================

This file summarizes the performance metrics for all trained classification models.

---
## Baseline Model Performance
---

### --- Logistic Regression ---
Accuracy: 0.85
AUC Score: 0.90
Precision (Class 1): 0.83
Recall (Class 1): 0.86
F1-score (Class 1): 0.84

### --- Decision Tree ---
Accuracy: 0.79
AUC Score: 0.79
Precision (Class 1): 0.78
Recall (Class 1): 0.79
F1-score (Class 1): 0.78

### --- Random Forest ---
Accuracy: 0.87
AUC Score: 0.92
Precision (Class 1): 0.86
Recall (Class 1): 0.88
F1-score (Class 1): 0.87

### --- Support Vector Machine (SVM) ---
Accuracy: 0.85
AUC Score: 0.91
Precision (Class 1): 0.84
Recall (Class 1): 0.86
F1-score (Class 1): 0.85


---
## Optimized Model Performance (After Hyperparameter Tuning)
---

### --- Tuned Random Forest (Best Model) ---
Accuracy: 0.90
AUC Score: 0.94
Precision (Class 1): 0.91
Recall (Class 1): 0.89
F1-score (Class 1): 0.90


---
## Final Selection
---
The **Tuned Random Forest** model was selected for deployment due to its superior performance across all key metrics.