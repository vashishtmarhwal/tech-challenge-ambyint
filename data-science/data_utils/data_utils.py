import polars as pl
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.preprocessing import RobustScaler

def clean_and_convert_types(df: pl.DataFrame) -> pl.DataFrame:
    """Lowercase column names and converts schema to expected on data type"""
    
    # Standardize column names to lowercase
    renamed_cols = {
        col: col.lower() 
        for col in df.columns
    }
    df = df.rename(renamed_cols)
    
    # Convert Timestamp to datetime
    # The schema indicates 'Timestamp' is a String, so parsing is needed.
    if 'timestamp' in df.columns:
        try:
            if not isinstance(df["timestamp"][0], pl.datatypes.Datetime):
                 df = df.with_columns(pl.col("timestamp").str.to_datetime(strict=False))
        except Exception as e:
            print(f"Error converting 'timestamp' column to datetime: {e}")
            raise
    else:
        raise KeyError("'timestamp' column (after lowercasing) not found.")
        
    return df

def missing_values_analysis(df: pl.DataFrame) -> pl.DataFrame:
    """Returns a DataFrame with the count of missing values per column."""
    
    total_rows = df.height
    
    # Calculate missing count and percentage per column
    missing_summary = pl.DataFrame([
        {
            "column": col,
            "dtype": str(df.schema[col]),
            "missing_count": df[col].null_count(),
            "missing_pct": df[col].null_count() / total_rows * 100
        }
        for col in df.columns
    ])

    missing_summary = (
        missing_summary
        .filter(pl.col("missing_pct") > 0)
        .sort("missing_pct", descending=True)
    )
    
    return missing_summary

def drop_columns(df: pl.DataFrame, columns: list) -> pl.DataFrame:
    """Drops specified columns from the DataFrame."""
    
    # Ensure columns to drop are in the DataFrame
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
    
    return df.drop(columns)

def handle_missing_values(df: pl.DataFrame, imputation_map: dict) -> pl.DataFrame:
    """Handles missing values in the DataFrame based on a predefined strategy."""
    
    df_processed = df.clone()

    # Apply imputations
    expressions_to_apply = []
    for col_name, expression in imputation_map.items():
        if col_name in df_processed.columns:
            print(f"Applying imputation for: {col_name}")
            expressions_to_apply.append(expression)
        else:
            print(f"Column '{col_name}' not found for imputation.")

    if expressions_to_apply:
        df_processed = df_processed.with_columns(expressions_to_apply)

    return df_processed

def create_time_features(df: pl.DataFrame, timestamp_col: str = "timestamp") -> pl.DataFrame:
    """Creates time-based features from a specified timestamp column in a Polars DataFrame."""

    if timestamp_col not in df.columns:
        print(f"Timestamp column '{timestamp_col}' not found in DataFrame.")
        return df

    if not isinstance(df.schema[timestamp_col], pl.Datetime):
        print(f"Column '{timestamp_col}' is not of Polars Datetime type. Found: {df.schema[timestamp_col]}")
        print("Please ensure the column is cast to pl.Datetime, e.g., df.with_columns(pl.col('your_col').str.to_datetime())")
        return df

    print(f"Creating time-based features from column: {timestamp_col}")
    
    df_with_time_features = df.with_columns([
        pl.col(timestamp_col).dt.year().alias(f"{timestamp_col}_year"),
        pl.col(timestamp_col).dt.month().alias(f"{timestamp_col}_month"),
        pl.col(timestamp_col).dt.day().alias(f"{timestamp_col}_day_of_month"),
        pl.col(timestamp_col).dt.weekday().alias(f"{timestamp_col}_day_of_week"), # Monday=1, Sunday=7
        pl.col(timestamp_col).dt.ordinal_day().alias(f"{timestamp_col}_day_of_year"), # Day of the year (1-366)
        pl.col(timestamp_col).dt.hour().alias(f"{timestamp_col}_hour"),
        pl.col(timestamp_col).dt.minute().alias(f"{timestamp_col}_minute"),
        pl.col(timestamp_col).dt.week().alias(f"{timestamp_col}_week_of_year"),
        pl.col(timestamp_col).dt.quarter().alias(f"{timestamp_col}_quarter"),
    ])

    print("Time-based features created successfully.")
    return df_with_time_features.drop(timestamp_col)  # Optionally drop the original timestamp column

def one_hot_encode_categorical(
    df: pl.DataFrame,
    categorical_columns: list
) -> pl.DataFrame:
    """Performs one-hot encoding on specified categorical columns in a Polars DataFrame."""

    if not categorical_columns:
        print("No categorical columns specified for encoding. Returning original DataFrame.")
        return df

    existing_categorical_columns = [col for col in categorical_columns if col in df.columns]
    if not existing_categorical_columns:
        print("None of the specified categorical columns exist in the DataFrame. Returning original DataFrame.")
        return df
        
    print(f"Performing one-hot encoding for columns: {existing_categorical_columns}")

    try:
        df_encoded = df.to_dummies(columns=existing_categorical_columns)
        print("One-hot encoding completed successfully.")
    except Exception as e:
        print(f"Error during one-hot encoding: {e}")
        print("Ensure columns are of appropriate types (String, Int, Bool, Category).")
        return df # Return original df on error
        
    return df_encoded


def scale_features_sklearn_robust(
    df: pl.DataFrame,
    columns_to_scale: List[str],
    scaler_object: Optional[RobustScaler] = None
) -> Tuple[pl.DataFrame, RobustScaler]:
    """
    Scales specified numerical features in a Polars DataFrame using
    scikit-learn's RobustScaler.

    If 'scaler_object' is None (typically for training data):
        Initializes and fits a new RobustScaler on the specified columns,
        applies scaling, and returns the scaled df and the fitted scaler_object.
    If 'scaler_object' is provided (typically for validation/test data):
        Uses the provided RobustScaler to scale the df and returns the
        scaled df and the original scaler_object.

    Args:
        df: Input Polars DataFrame.
        columns_to_scale: List of numerical column names to scale. These columns
                          should be free of NaNs for predictable behavior,
                          although RobustScaler has some NaN handling during fit.
        scaler_object: Optional pre-fitted scikit-learn RobustScaler object.

    Returns:
        A tuple containing:
            - df_processed: Polars DataFrame with specified columns scaled.
            - learned_scaler_object: The scikit-learn RobustScaler object
                                     (either newly fitted or the one passed in).
    """
    df_processed = df.clone()
    is_fitting_mode = scaler_object is None

    actual_columns_to_scale = [col for col in columns_to_scale if col in df_processed.columns]
    
    # Filter for numeric types only from the actual columns to scale
    numeric_cols_for_scaling = []
    for col_name in actual_columns_to_scale:
        if df_processed[col_name].dtype in pl.NUMERIC_DTYPES:
            numeric_cols_for_scaling.append(col_name)
        else:
            print(f"Warning: Column '{col_name}' is of non-numeric type "
                  f"{df_processed[col_name].dtype}. Skipping scaling.")
    
    if not numeric_cols_for_scaling:
        print("No valid numeric columns found to scale. Returning original DataFrame and initial scaler object.")
        # If in fitting mode and no valid columns, return a new unfitted scaler
        return df_processed, scaler_object if scaler_object else RobustScaler()

    # Extract data to be scaled as a NumPy array
    # Ensure no all-null columns are passed to .to_numpy() if they cause issues,
    # though RobustScaler might handle them by ignoring.
    # It's better if imputation has already handled extensive NaNs.
    data_to_scale_np = df_processed.select(numeric_cols_for_scaling).to_numpy()

    # Check for columns that are entirely NaN after conversion to numpy,
    # as this can cause issues with RobustScaler fit if not handled.
    # RobustScaler can sometimes ignore them if a feature is all NaN during fit.
    if np.all(np.isnan(data_to_scale_np), axis=0).any() and is_fitting_mode:
        print("Warning: One or more columns to scale are entirely NaN. "
              "RobustScaler might ignore these during fit. Ensure imputation is complete.")

    if is_fitting_mode:
        current_scaler = RobustScaler()
        try:
            scaled_data_np = current_scaler.fit_transform(data_to_scale_np)
        except ValueError as e:
            print(f"Error during RobustScaler fit_transform: {e}. "
                  "This might be due to all-NaN columns or other data issues. "
                  "Returning original DataFrame.")
            return df_processed, current_scaler # Return original df and unfitted scaler
    else: # Transform mode
        if scaler_object is None:
            # This case should ideally not be reached if logic is followed,
            # but as a safeguard:
            print("Error: scaler_object is None in transform mode. Cannot proceed.")
            return df_processed, None # Or raise an error
        current_scaler = scaler_object
        try:
            scaled_data_np = current_scaler.transform(data_to_scale_np)
        except ValueError as e:
            print(f"Error during RobustScaler transform: {e}. "
                  "Ensure the scaler was fitted correctly and data is valid. "
                  "Returning original DataFrame.")
            return df_processed, current_scaler


    # Create Polars expressions to update columns with scaled data
    update_expressions = []
    for i, col_name in enumerate(numeric_cols_for_scaling):
        # Create a Polars Series from the scaled NumPy column
        # Handle potential all-NaN columns from scaling if necessary, though
        # RobustScaler usually outputs numbers or NaNs where input was NaN.
        scaled_series = pl.Series(name=col_name, values=scaled_data_np[:, i])
        update_expressions.append(scaled_series)

    if update_expressions:
        df_processed = df_processed.with_columns(update_expressions)
        print(f"Applied scikit-learn RobustScaler to columns: {numeric_cols_for_scaling}")
    else:
        print("No scaling expressions were applied (e.g., due to errors or no valid columns).")
        
    return df_processed, current_scaler