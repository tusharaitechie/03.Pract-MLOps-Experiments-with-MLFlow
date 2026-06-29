from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
import mlflow.sklearn
import os

# =============================================================================
# Load the Breast Cancer dataset
# =============================================================================
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# =============================================================================
# Split into training and testing sets
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =============================================================================
# Create the Random Forest model
# =============================================================================
rf = RandomForestClassifier(random_state=42)

# =============================================================================
# Define parameter grid for hyperparameter tuning
# =============================================================================
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [None, 10, 20, 30]
}

# =============================================================================
# GridSearchCV
# =============================================================================
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

# =============================================================================
# -------------------- Run WITHOUT MLflow --------------------
# Uncomment this block if you only want GridSearchCV results
# =============================================================================

# grid_search.fit(X_train, y_train)
#
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
#
# print(best_params)
# print(best_score)

# =============================================================================
# -------------------- MLflow Tracking --------------------
# =============================================================================

# Uncomment this only if MLflow Tracking Server is running
# Example:
# mlflow ui
#
# Then uncomment the line below.
#
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set experiment
mlflow.set_experiment("breast-cancer-rf-hp")

# Parent Run
with mlflow.start_run(run_name="GridSearchCV-RandomForest") as parent_run:

    # Train GridSearchCV
    grid_search.fit(X_train, y_train)

    # ==========================================================
    # Log every parameter combination as a child run
    # ==========================================================
    for i in range(len(grid_search.cv_results_["params"])):

        with mlflow.start_run(
            nested=True,
            run_name=f"trial_{i+1}"
        ):

            # Log parameters
            mlflow.log_params(grid_search.cv_results_["params"][i])

            # Log validation accuracy
            mlflow.log_metric(
                "accuracy",
                grid_search.cv_results_["mean_test_score"][i]
            )

    # ==========================================================
    # Best model information
    # ==========================================================
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log best parameters
    mlflow.log_params(best_params)

    # Log best score
    mlflow.log_metric("best_accuracy", best_score)

    # ==========================================================
    # Log Training Dataset
    # ==========================================================
    train_df = X_train.copy()
    train_df["target"] = y_train.astype("float64")

    train_dataset = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_dataset, context="training")

    # ==========================================================
    # Log Testing Dataset
    # ==========================================================
    test_df = X_test.copy()
    test_df["target"] = y_test.astype("float64")

    test_dataset = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_dataset, context="testing")

    # ==========================================================
    # Log source code
    # ==========================================================
    mlflow.log_artifact(os.path.abspath(__file__))

    # ==========================================================
    # Log best model
    # (Using new API - no deprecation warning)
    # ==========================================================
    mlflow.sklearn.log_model(
        sk_model=grid_search.best_estimator_,
        name="random_forest"
    )

    # ==========================================================
    # Tags
    # ==========================================================
    mlflow.set_tags({
        "author": "Tushar",
        "algorithm": "RandomForest",
        "dataset": "Breast Cancer",
        "task": "Hyperparameter Tuning"
    })

    # ==========================================================
    # Print Results
    # ==========================================================
    print("\nBest Parameters:")
    print(best_params)

    print("\nBest Accuracy:")
    print(best_score)