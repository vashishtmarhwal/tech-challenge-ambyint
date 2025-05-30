from typing import List, Any, Optional

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score,
    roc_curve
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, StratifiedKFold


def evaluate_model(
    y_val: np.ndarray,
    y_pred_test_rf: np.ndarray,
    y_proba_test_rf: np.ndarray = None,
    plot_cm: bool = False,
    plot_roc_curve: bool = False
):
    """Evaluates the performance of a model using various metrics."""
    
    try:
        accuracy_rf = accuracy_score(y_val, y_pred_test_rf)
        precision_rf = precision_score(y_val, y_pred_test_rf, zero_division=0)
        recall_rf = recall_score(y_val, y_pred_test_rf, zero_division=0)
        f1_rf = f1_score(y_val, y_pred_test_rf, zero_division=0)
        if y_proba_test_rf is not None:
            roc_auc_rf = roc_auc_score(y_val, y_proba_test_rf)

        print(f"Accuracy:  {accuracy_rf:.4f}")
        print(f"Precision (for class 1): {precision_rf:.4f}") # How many selected items are relevant?
        print(f"Recall (for class 1):    {recall_rf:.4f}") # How many relevant items are selected? (Sensitivity)
        print(f"F1-score (for class 1):  {f1_rf:.4f}")
        if y_proba_test_rf is not None:
            print(f"ROC AUC Score: {roc_auc_rf:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_val, y_pred_test_rf, zero_division=0))

        print("\nConfusion Matrix:")
        cm_rf = confusion_matrix(y_val, y_pred_test_rf)
        print(cm_rf)

        if plot_cm:
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Predicted 0', 'Predicted 1'],
                        yticklabels=['Actual 0', 'Actual 1'])
            plt.title('Confusion Matrix - Random Forest')
            plt.ylabel('Actual Class')
            plt.xlabel('Predicted Class')
            plt.show()

        if plot_roc_curve and y_proba_test_rf is not None:
            fpr, tpr, _ = roc_curve(y_val, y_proba_test_rf)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc_rf:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title('ROC Curve - Random Forest')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.show()


    except Exception as e:
        print(f"Error during evaluation: {e}")


def plot_tree_based_feature_importances(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
    importance_type_name: str = "Importance Score",
    figsize: tuple = (10, 7),
    color: str = 'skyblue'
):
    """Plots top N feature importances from a model with a `feature_importances_` attribute."""
    importances = model.feature_importances_
    actual_top_n = min(top_n, len(importances))
    # Sort features by importance
    sorted_indices = np.argsort(importances)
    top_indices = sorted_indices[-actual_top_n:]

    plt.figure(figsize=figsize)
    plt.barh(range(actual_top_n), importances[top_indices], color=color, align='center')
    plt.yticks(range(actual_top_n), [feature_names[i] for i in top_indices])
    plt.xlabel(importance_type_name)
    plt.title(f"Top {actual_top_n} Feature Importances ({importance_type_name})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_permutation_feature_importances(
    model: Any, # Any fitted scikit-learn compatible model
    X_val: pd.DataFrame,
    y_val: Any, # pd.Series or np.array
    top_n: int = 20,
    n_repeats: int = 10,
    scoring: Optional[str] = None,
    random_state: Optional[int] = 42,
    n_jobs: Optional[int] = -1,
    figsize: tuple = (10, 7),
    color: str = 'salmon'
):
    """Calculates and plots top N permutation importances."""
    feature_names = X_val.columns.tolist()
    try:
        result = permutation_importance(
            model, X_val, y_val,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs
        )
    except Exception as e:
        print(f"Error calculating permutation importance: {e}")
        return

    importances_mean = result.importances_mean
    actual_top_n = min(top_n, len(importances_mean))
    # Sort features by importance
    sorted_indices = np.argsort(importances_mean)
    top_indices = sorted_indices[-actual_top_n:]

    plt.figure(figsize=figsize)
    plt.barh(range(actual_top_n), importances_mean[top_indices], color=color, align='center')
    plt.yticks(range(actual_top_n), [feature_names[i] for i in top_indices])
    plt.xlabel(f"Permutation Importance ({scoring if scoring else 'default scorer'})")
    plt.title(f"Top {actual_top_n} Permutation Importances")
    plt.gca().invert_yaxis() # Display most important at the top
    plt.tight_layout()
    plt.show()


def run_feature_importance_analysis(
    model: Any,
    X_val: Any, 
    y_val: Any,
    feature_names_list: Optional[List[str]] = None,
    top_n: int = 20,
    plot_tree_importances: bool = True,
    tree_importance_type_name: str = "Tree-based Importance",
    plot_permutation_importances: bool = True,
    perm_n_repeats: int = 10,
    perm_scoring: Optional[str] = None,
    perm_random_state: Optional[int] = 42,
    perm_n_jobs: Optional[int] = -1
):
    """Runs and plots different types of feature importances for a model."""
    X_val_df = None
    processed_feature_names = None

    if isinstance(X_val, pd.DataFrame):
        X_val_df = X_val
        processed_feature_names = X_val_df.columns.tolist()
    elif isinstance(X_val, np.ndarray):
        if feature_names_list:
            if len(feature_names_list) == X_val.shape[1]:
                processed_feature_names = feature_names_list
                X_val_df = pd.DataFrame(X_val, columns=processed_feature_names) # Needed for perm. importance
            else:
                print("Error: Length of feature_names_list does not match X_val columns. Cannot proceed.")
                return
        else:
            # Create generic feature names for NumPy array if none provided
            processed_feature_names = [f"feature_{i}" for i in range(X_val.shape[1])]
            X_val_df = pd.DataFrame(X_val, columns=processed_feature_names) # Needed for perm. importance
            if plot_tree_importances and hasattr(model, 'feature_importances_'):
                print("Warning: X_val is a NumPy array and feature_names_list not provided. "
                      "Tree-based importances will use generic labels.")
    else:
        print(f"Error: X_val is of unsupported type {type(X_val)}. Please provide Pandas DataFrame or NumPy array.")
        return

    if plot_tree_importances:
        print(f"\nPlotting {tree_importance_type_name}...")
        if processed_feature_names:
            plot_tree_based_feature_importances(
                model,
                feature_names=processed_feature_names,
                top_n=top_n,
                importance_type_name=tree_importance_type_name
            )
        else: 
            print("Skipping tree-based feature importances: feature names could not be determined.")
        
    if plot_permutation_importances:
        print("\nCalculating and Plotting Permutation Importances...")
        if X_val_df is not None: 
            plot_permutation_feature_importances(
                model,
                X_val=X_val_df, 
                y_val=y_val,
                top_n=top_n,
                n_repeats=perm_n_repeats,
                scoring=perm_scoring,
                random_state=perm_random_state,
                n_jobs=perm_n_jobs
            )
        else:
            print("Skipping permutation importances as a suitable DataFrame for X_val could not be constructed.")


def perform_cross_validation(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[Any] = None, 
    scoring: str = 'f1',
    n_splits: int = 5,
) -> np.ndarray:
    """Performs cross-validation and returns scores."""
    
    if cv is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        print(f"Cross-validation {scoring} scores:", scores)
        print(f"Mean {scoring} score: {np.mean(scores):.4f}")
        print(f"Standard deviation: {np.std(scores):.4f}")
        return scores
    except Exception as e:
        print(f"Error during cross-validation: {e}")
        return np.array([])