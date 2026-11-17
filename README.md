# Telecom Plan Recommendation: Classification Model

## Project Overview

A mobile operator wants to recommend the most suitable of two plans to its subscribers based on how they use their phones. This project frames the task as binary classification: predicting whether a subscriber belongs on the Ultra plan (`is_ultra = 1`) from four usage features (calls, minutes, messages and data traffic). Three algorithms, Random Forest, Logistic Regression and Decision Tree, are compared against a target accuracy of 0.75.

**Key outcome:** a Random Forest with 16 trees reached **0.802 validation accuracy**, the only model above the 0.75 threshold.

## Business Problem

Subscribers whose plan does not match their usage may pay more than necessary or miss features they would use. An automated recommendation based on actual behavior lets the operator suggest a suitable plan to each customer at scale.

## Objectives

- Build a classifier that predicts the appropriate plan from monthly usage.
- Reach an accuracy of at least **0.75** (stored as `acc_limit` in the notebook).
- Compare several algorithms and tune their key hyperparameters.

## Dataset Description

`users_behavior.csv`, provided for educational purposes as part of the TripleTen Data Science program, contains 3,214 rows (one per subscriber) and 5 columns, with no missing values.

| Column | Type | Description |
|---|---|---|
| `calls` | float | Number of calls |
| `minutes` | float | Total call duration in minutes |
| `messages` | float | Number of text messages |
| `mb_used` | float | Internet traffic in MB |
| `is_ultra` | int | **Target:** 1 = Ultra plan, 0 = other plan |

## Technologies and Libraries

- Python
- pandas
- scikit-learn (`RandomForestClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `train_test_split`, `accuracy_score`)
- Jupyter Notebook

## Methodology

1. **Data split:** 75% training and 25% validation (`random_state=54321`). All features are numeric, so no encoding is needed.
2. **Random Forest:** `n_estimators` is tuned from 1 to 19, keeping the value with the highest validation accuracy.
3. **Logistic Regression:** trained with the `liblinear` solver as a linear baseline; accuracy is reported on both the training and validation sets to check for under- or overfitting.
4. **Decision Tree:** `max_depth` values from 1 to 19 are evaluated on the validation set.
5. **Metric:** accuracy, which matches the business requirement.

## Results and Model Performance

| Model | Configuration | Training accuracy | Validation accuracy |
|---|---|---|---|
| **Random Forest** | `n_estimators = 16` (best of 1–19) | — | **0.802** |
| Decision Tree | `max_depth = 19` (see limitations) | — | 0.731 |
| Logistic Regression | `solver='liblinear'` | 0.718 | 0.694 |

Target: accuracy ≥ 0.75. No separate test set was used; all values in the validation column come from the same validation split.

## Key Findings

- The Random Forest clearly outperforms the other models and is the only one that meets the accuracy requirement.
- Logistic Regression stays below the threshold even on the training data (0.718), which indicates underfitting: a linear decision boundary on unscaled usage features is not flexible enough for this problem.
- A small ensemble (16 trees) is enough to exceed the threshold, which keeps the model lightweight.

## Conclusions

**Achievement of objectives:** the Random Forest classifier meets the 0.75 accuracy target on the validation set and is the recommended model.

**Limitations**

- There is no held-out test set: hyperparameter tuning and model comparison both use the same validation set, so the 0.802 accuracy is an optimistic estimate.
- In the Decision Tree loop, `best_acc` and `best_depth` are reset on every iteration, so the reported result corresponds to the last depth evaluated (19) rather than to the best one. The best Decision Tree accuracy is therefore not known from this run.
- Some printed labels in the saved outputs differ slightly from the current code (for example `"MODELO UTILIZADO = "` vs. `"MODELO UTILIZADO - "`), which indicates that cells were edited after execution. The metrics themselves are unaffected.
- The class balance of `is_ultra` was not examined, and accuracy was the only metric reported.
- The features were not scaled for Logistic Regression.

**Potential improvements:** split the data into training, validation and test sets; fix the Decision Tree search loop; tune more Random Forest hyperparameters (for example `max_depth`); compare against a majority-class baseline; and report precision, recall and F1 in addition to accuracy.

## Repository Structure

```text
telecom-plan-recommendation-classification/
├── README.md
├── main.ipynb
└── datasets/
    └── users_behavior.csv
```

## How to Run

1. Create and activate a Python 3 environment (the notebook was last run with Python 3.13):

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. Install the dependencies:

   ```bash
   pip install pandas scikit-learn jupyter
   ```

3. Start Jupyter **from the project root**, so that the relative path `./datasets/users_behavior.csv` resolves correctly, and open `main.ipynb`:

   ```bash
   jupyter notebook main.ipynb
   ```

Printed messages are in Portuguese, as in the original run.

## Author

**Luiz Otávio Carneiro Filho**
Geospatial Engineer | GIS & Data Science
[LinkedIn](https://www.linkedin.com/in/luizot/)
