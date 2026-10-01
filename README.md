# Student Math Score Prediction — Model Training

A machine learning project that predicts a student's **math score** from demographic, educational, and test-preparation information. The notebook compares multiple regression algorithms, evaluates their performance using standard regression metrics, and visualizes actual vs. predicted scores.

## Project Overview

The goal of this project is to build and compare regression models for predicting `math_score`.

The workflow includes:

1. Loading the student performance dataset.
2. Separating features (`X`) and the target (`y`).
3. Identifying numerical and categorical features.
4. Encoding categorical variables with One-Hot Encoding.
5. Standardizing numerical variables with `StandardScaler`.
6. Splitting the data into training and testing sets.
7. Training and comparing several regression models.
8. Evaluating models using MAE, RMSE, and R².
9. Inspecting predictions from Linear Regression.
10. Visualizing actual vs. predicted values.
11. Demonstrating a pipeline-based approach that performs preprocessing using the training data only.

## Models Used

The notebook compares the following regression algorithms:

- Linear Regression
- Lasso Regression
- Ridge Regression
- K-Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor
- Support Vector Regression (`SVR`) is imported in the notebook but is not included in the model comparison dictionary.

## Dataset

The notebook expects a CSV file named:

```text
stud.csv
```

The target column is:

```text
math_score
```

The feature columns used by the notebook are all remaining columns in the dataset. The categorical variables explicitly inspected in the notebook include:

- `gender`
- `race_ethnicity`
- `parental_level_of_education`
- `lunch`
- `test_preparation_course`

The dataset is expected to contain the corresponding student-performance columns.

> **Note:** The dataset itself is not included in this repository. Place `stud.csv` in the project directory before running the notebook, or update the CSV path in the notebook.

## Preprocessing

### Categorical Features

Categorical columns are transformed using:

```python
OneHotEncoder()
```

This converts categorical values into numerical indicator variables suitable for machine learning models.

### Numerical Features

Numerical columns are standardized using:

```python
StandardScaler()
```

### Train/Test Split

The data is divided into:

- **80% training data**
- **20% testing data**

with:

```python
random_state=42
```

This makes the split reproducible.

## Model Evaluation

Each model is evaluated using three regression metrics:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

**Lower is better.**

### Root Mean Squared Error (RMSE)

Measures the square root of the average squared prediction error. Because larger errors are penalized more heavily, RMSE is useful for identifying models that make large mistakes.

**Lower is better.**

### R² Score

Measures how much of the variation in the target variable is explained by the model.

A value closer to `1` generally indicates a better fit.

**Higher is better.**

The notebook creates a model-comparison table containing the test-set R² score for each model.

## Results

The notebook evaluates the models programmatically rather than hard-coding a single "best" model in this README.

To see the actual results, run the notebook and inspect the **Results** section, which sorts the models by test-set R² score.

This approach keeps the README reproducible and avoids reporting results that may change with library versions, dataset changes, or preprocessing changes.

## Visualizations

The notebook includes:

- Actual vs. predicted scatter plot
- Regression plot comparing actual and predicted values
- A table showing actual values, predicted values, and prediction differences

These visualizations help assess how closely the predictions follow the actual math scores.

## Project Structure

A recommended repository structure is:

```text
student-math-score-prediction/
│
├── 2. MODEL TRAINING.ipynb
├── stud.csv
├── README.md
└── requirements.txt
```

If the dataset is not meant to be committed to GitHub, keep it outside the repository and update the notebook's data-loading path accordingly.

## Installation

Clone the repository and move into the project directory:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
2. MODEL TRAINING.ipynb
```

Make sure `stud.csv` is available at the path expected by the notebook.

## Important Implementation Note

The notebook initially preprocesses the complete feature dataset before performing the train/test split:

```python
X = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(...)
```

For a rigorous machine-learning workflow, preprocessing should be **fit only on the training data**. Otherwise, information from the test set can influence the preprocessing step.

The notebook later demonstrates the preferred approach using a `Pipeline`:

```python
Pipeline([
    ("prep", preprocessor),
    ("model", model)
])
```

This allows the preprocessing transformations to be learned from the training data and then applied to the test data.

For a production-quality version of the project, the pipeline-based approach should be used consistently throughout model training and evaluation.

## Future Improvements

Possible extensions to this project include:

- Hyperparameter tuning with `RandomizedSearchCV` or `GridSearchCV`
- Cross-validation for more reliable model comparison
- Feature importance analysis
- Residual/error analysis
- Saving the trained model with `joblib`
- Building a prediction interface using Streamlit or Flask
- Adding automated data validation
- Tracking experiments and model versions
- Creating a cleaner end-to-end training pipeline

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- CatBoost
- Jupyter Notebook

## Author

**Sanyam Jha**

This project was created as part of a practical machine learning/data science workflow for learning regression, preprocessing, model comparison, and evaluation.
