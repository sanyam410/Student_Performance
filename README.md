# Student Math Score Prediction: Comparing 9 Regression Models

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-regression-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)

Predicting a student's **math score** from demographic, educational and test-preparation attributes, and finding out which of nine regression algorithms generalises best to unseen students.

![Model comparison](images/model_comparison.png)

## Key Findings

| | Finding |
|---|---|
| **Best models** | **Ridge** (test R² = 0.8806) and **Linear Regression** (0.8804). The gap between them is 0.0002, which is effectively a tie. |
| **Typical error** | The best model is off by about **4.2 marks on average** (MAE) and about **5.4 marks** in RMSE terms, on a 0-100 scale. |
| **Simple beat complex** | Every tree-based or boosted model scored lower on the test set than plain linear regression. XGBoost reached 0.8278 against 0.8806 for Ridge. |
| **Overfitting is clear** | The Decision Tree scored **0.9997 on training data but 0.7277 on test data**, a gap of about 0.27. |

The main lesson: on a small tabular dataset where the signal is mostly additive, a regularised linear model can beat far more flexible algorithms, and the train/test gap is what exposes the difference.

## Problem and Dataset

- **Task:** supervised regression, predicting `math_score`.
- **Data:** a student performance dataset of about 1,000 records (800 train / 200 test after the split), loaded from `stud.csv`.
- **Categorical features inspected:** `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`.
- **Target:** `math_score`; all remaining columns are used as features.

> **Note:** the dataset is not included in this repository. Place `stud.csv` in the project directory (or update the path in the notebook) before running it.

## Approach

```
stud.csv → split features / target → One-Hot Encode categoricals
         → Standardise numericals → 80/20 train-test split (random_state=42)
         → train 9 models → evaluate on MAE, RMSE, R² → compare train vs test
```

1. **Preprocessing** with a `ColumnTransformer`: `OneHotEncoder` for categorical columns and `StandardScaler` for numerical columns.
2. **Models compared:** Linear Regression, Lasso, Ridge, K-Neighbors, Decision Tree, Random Forest, XGBoost, CatBoost, AdaBoost, all with default hyperparameters. (`SVR` is imported but not part of the comparison.)
3. **Evaluation** on three metrics, computed on both training and test sets so overfitting is visible:
   - **MAE**: average absolute error in marks (lower is better)
   - **RMSE**: like MAE but penalises large misses more heavily (lower is better)
   - **R²**: share of score variation the model explains (higher is better)
4. **Diagnostics:** actual vs. predicted scatter plot, regression plot, and a residual table for the best linear model.
5. **Pipeline version:** the notebook also demonstrates a `Pipeline` that fits preprocessing on the training data only (see [Limitations](#limitations-and-honest-caveats)).

## Results

All figures are on the **held-out test set** unless stated otherwise. Models are ranked by test R².

| Rank | Model | Test R² | Test RMSE | Test MAE | Train R² | Train − Test R² |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Ridge** | **0.8806** | **5.3904** | **4.2111** | 0.8743 | −0.0063 |
| 2 | **Linear Regression** | 0.8804 | 5.3940 | 4.2148 | 0.8743 | −0.0061 |
| 3 | AdaBoost | 0.8556 | 5.9275 | 4.6135 | 0.8548 | −0.0008 |
| 4 | Random Forest | 0.8522 | 5.9976 | 4.6359 | 0.9767 | 0.1245 |
| 5 | CatBoost | 0.8516 | 6.0086 | 4.6125 | 0.9589 | 0.1073 |
| 6 | XGBoost | 0.8278 | 6.4733 | 5.0577 | 0.9955 | 0.1677 |
| 7 | Lasso | 0.8253 | 6.5197 | 5.1579 | 0.8071 | −0.0182 |
| 8 | K-Neighbors | 0.7838 | 7.2530 | 5.6210 | 0.8555 | 0.0717 |
| 9 | Decision Tree | 0.7277 | 8.1406 | 6.5400 | 0.9997 | 0.2720 |

*A small negative gap means the test score was marginally higher than the training score. With only 200 test rows this is sampling noise, not a real advantage.*

## What the Results Show

**1. Linear models win, and Ridge and Linear Regression are tied.**
Their test R² values differ by 0.0002, far smaller than the noise from a single 200-row test set. I would not claim Ridge is "better"; the honest reading is that regularisation adds essentially nothing here, which suggests the linear model is not overfitting in the first place.

**2. Flexible models memorise the training data.**
Random Forest, XGBoost, CatBoost and the Decision Tree all reach 0.96-0.9997 R² on training data but drop to 0.73-0.85 on test data. The linear models and AdaBoost show almost no gap. With roughly 800 training rows, models with high capacity have enough freedom to fit noise.

**3. The Decision Tree is the clearest overfitting example.**
Its training RMSE is 0.28 marks (near-perfect recall of the training set) against a test RMSE of 8.14. An unpruned tree with default settings will keep splitting until it fits every training point.

**4. Lasso likely underperformed because it was untuned.**
At 0.8253 it trails Ridge by about 0.055 R² and has the lowest training score of all models (0.8071). That pattern is consistent with the default regularisation strength shrinking coefficients too aggressively, but I have not run a tuning search to confirm it.

**5. Errors are larger at the extremes.**
In the residual table, a student who actually scored 91 was predicted at about 76, an error of roughly 14.6 marks, while mid-range students were typically predicted within a few marks. This is an observation from the sample rows shown, not a measured trend, but it is the classic behaviour of regression models pulling predictions toward the mean.

## Limitations and Honest Caveats

- **Preprocessing leakage in the main comparison.** The main loop calls `preprocessor.fit_transform(X)` *before* the train/test split, so the scaler and encoder learn statistics from rows that later become test data. The effect on this dataset should be small, but it is not rigorous. The notebook includes a `Pipeline` version that fits preprocessing on training data only, and that is the approach to use in any real workflow.
- **Single train/test split.** Rankings come from one 80/20 split with `random_state=42`. Without cross-validation, small differences between models (Ridge vs. Linear Regression, or Random Forest vs. CatBoost) should not be treated as meaningful.
- **No hyperparameter tuning.** Every model uses default settings, so tree-based and boosted models may be under-served here. Tuning could narrow the gap, though their overfitting pattern suggests they would need strong regularisation.
- **Predictive exercise only.** The model uses demographic attributes such as gender and parental education. It is a learning project on regression technique and should not be used to make decisions about real students.

## Reproducing the Results

```bash
git clone <your-repository-url>
cd <your-repository-folder>

python -m venv .venv
# Windows:       .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Open `2. MODEL TRAINING.ipynb`, make sure `stud.csv` is available at the path the notebook expects, and run all cells. Exact metric values can shift slightly with library versions.

## Project Structure

```text
student-math-score-prediction/
├── 2. MODEL TRAINING.ipynb     # preprocessing, model comparison, diagnostics
├── images/
│   └── model_comparison.png    # results chart used in this README
├── stud.csv                    # dataset (add locally; not committed)
├── requirements.txt
└── README.md
```

## Roadmap

In rough priority order:

1. **Use the `Pipeline` for every model**, so the leakage caveat disappears from the main results.
2. **Add k-fold cross-validation** to report mean ± standard deviation of R² instead of a single split.
3. **Tune hyperparameters** with `RandomizedSearchCV`, starting with Lasso's alpha and the tree-based models' depth and regularisation.
4. **Residual analysis** to test whether errors really grow at the score extremes.
5. **Feature importance / coefficient analysis** to explain which attributes drive predictions.
6. **Package the best model** with `joblib` and expose it through a small Streamlit or Flask app.

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost · CatBoost · Jupyter Notebook

## Author

**Sanyam Jha**

Created as a hands-on exercise in regression, preprocessing, model comparison and honest evaluation.
