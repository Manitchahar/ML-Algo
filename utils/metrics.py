"""
Evaluation metrics for machine learning algorithms.

This module provides implementations of common evaluation metrics
used to assess the performance of machine learning models.

Author: ML-Algo Contributors
Date: 2025
License: MIT
"""

import numpy as np
from typing import Union


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error between true and predicted values.
    
    Mean Squared Error (MSE) measures the average of the squares of the errors.
    It is one of the most commonly used regression metrics.
    
    Mathematical Formula:
    MSE = (1/n) * Σ(y_true - y_pred)²
    
    Where:
    - n: number of samples
    - y_true: actual target values
    - y_pred: predicted target values
    
    Properties:
    - Always non-negative (≥ 0)
    - Perfect predictions yield MSE = 0
    - Heavily penalizes large errors due to squaring
    - Units: square of the target variable units
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True target values (ground truth)
    y_pred : array-like, shape (n_samples,)
        Predicted target values
    
    Returns:
    --------
    mse : float
        Mean squared error value
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from utils.metrics import mean_squared_error
    >>> 
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    >>> mse = mean_squared_error(y_true, y_pred)
    >>> print(f"MSE: {mse:.4f}")
    MSE: 0.0260
    
    Notes:
    ------
    - MSE is sensitive to outliers due to squaring of errors
    - Consider using MAE for robustness to outliers
    - RMSE (square root of MSE) is in the same units as the target variable
    """
    # Convert inputs to numpy arrays and validate
    y_true, y_pred = _validate_regression_targets(y_true, y_pred)
    
    # Calculate squared differences
    squared_errors = (y_true - y_pred) ** 2
    
    # Return mean of squared errors
    return np.mean(squared_errors)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Error between true and predicted values.
    
    Mean Absolute Error (MAE) measures the average of the absolute differences
    between predicted and actual values. It is more robust to outliers than MSE.
    
    Mathematical Formula:
    MAE = (1/n) * Σ|y_true - y_pred|
    
    Properties:
    - Always non-negative (≥ 0)
    - Perfect predictions yield MAE = 0
    - Linear penalty for errors (vs quadratic for MSE)
    - More robust to outliers than MSE
    - Units: same as the target variable units
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True target values (ground truth)
    y_pred : array-like, shape (n_samples,)
        Predicted target values
    
    Returns:
    --------
    mae : float
        Mean absolute error value
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from utils.metrics import mean_absolute_error
    >>> 
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    >>> mae = mean_absolute_error(y_true, y_pred)
    >>> print(f"MAE: {mae:.4f}")
    MAE: 0.1400
    """
    # Convert inputs to numpy arrays and validate
    y_true, y_pred = _validate_regression_targets(y_true, y_pred)
    
    # Calculate absolute differences
    absolute_errors = np.abs(y_true - y_pred)
    
    # Return mean of absolute errors
    return np.mean(absolute_errors)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error between true and predicted values.
    
    Root Mean Squared Error (RMSE) is the square root of MSE. It has the
    advantage of being in the same units as the target variable.
    
    Mathematical Formula:
    RMSE = √(MSE) = √((1/n) * Σ(y_true - y_pred)²)
    
    Properties:
    - Always non-negative (≥ 0)
    - Perfect predictions yield RMSE = 0
    - Same units as the target variable
    - Penalizes large errors more than MAE
    - More interpretable than MSE due to unit consistency
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True target values (ground truth)
    y_pred : array-like, shape (n_samples,)
        Predicted target values
    
    Returns:
    --------
    rmse : float
        Root mean squared error value
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from utils.metrics import root_mean_squared_error
    >>> 
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    >>> rmse = root_mean_squared_error(y_true, y_pred)
    >>> print(f"RMSE: {rmse:.4f}")
    RMSE: 0.1612
    """
    # Calculate MSE and take square root
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the R² (coefficient of determination) score.
    
    R² represents the proportion of the variance in the dependent variable
    that is predictable from the independent variables. It provides a measure
    of how well observed outcomes are replicated by the model.
    
    Mathematical Formula:
    R² = 1 - (SS_res / SS_tot)
    
    Where:
    - SS_res = Σ(y_true - y_pred)² (residual sum of squares)
    - SS_tot = Σ(y_true - y_mean)² (total sum of squares)
    - y_mean = mean of y_true
    
    Interpretation:
    - R² = 1: Perfect prediction (model explains 100% of variance)
    - R² = 0: Model performs as well as simply predicting the mean
    - R² < 0: Model performs worse than predicting the mean
    - R² ∈ (-∞, 1]: Theoretical range
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True target values (ground truth)
    y_pred : array-like, shape (n_samples,)
        Predicted target values
    
    Returns:
    --------
    r2 : float
        R² coefficient of determination
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from utils.metrics import r2_score
    >>> 
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    >>> r2 = r2_score(y_true, y_pred)
    >>> print(f"R² Score: {r2:.4f}")
    R² Score: 0.9870
    
    Notes:
    ------
    - R² can be negative for very poor models
    - For linear regression, R² equals the square of the correlation coefficient
    - Higher R² values indicate better model performance
    - Be cautious of overfitting with very high R² values on training data
    """
    # Convert inputs to numpy arrays and validate
    y_true, y_pred = _validate_regression_targets(y_true, y_pred)
    
    # Calculate residual sum of squares (SS_res)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calculate total sum of squares (SS_tot)
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    # Handle edge case where all true values are the same
    if ss_tot == 0:
        # If all y_true values are identical
        if ss_res == 0:
            # Perfect prediction when all values are the same
            return 1.0
        else:
            # Imperfect prediction when all true values are the same
            return 0.0
    
    # Calculate R² score
    r2 = 1 - (ss_res / ss_tot)
    return r2


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy classification score.
    
    Accuracy is the fraction of predictions that match the true class labels.
    It is the most intuitive performance measure for classification problems.
    
    Mathematical Formula:
    Accuracy = (Number of correct predictions) / (Total number of predictions)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Where:
    - TP: True Positives
    - TN: True Negatives  
    - FP: False Positives
    - FN: False Negatives
    
    Properties:
    - Range: [0, 1] where 1 is perfect accuracy
    - Symmetric: treats all classes equally
    - Can be misleading for imbalanced datasets
    - Simple and intuitive interpretation
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True class labels (ground truth)
    y_pred : array-like, shape (n_samples,)
        Predicted class labels
    
    Returns:
    --------
    accuracy : float
        Classification accuracy score between 0 and 1
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from utils.metrics import accuracy_score
    >>> 
    >>> y_true = np.array([0, 1, 2, 2, 1])
    >>> y_pred = np.array([0, 2, 1, 2, 1])
    >>> acc = accuracy_score(y_true, y_pred)
    >>> print(f"Accuracy: {acc:.4f}")
    Accuracy: 0.6000
    
    Notes:
    ------
    - Accuracy can be misleading for imbalanced datasets
    - Consider precision, recall, and F1-score for better evaluation
    - For multi-class problems, this gives overall accuracy across all classes
    """
    # Convert inputs to numpy arrays and validate
    y_true, y_pred = _validate_classification_targets(y_true, y_pred)
    
    # Calculate number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)
    
    # Calculate total number of predictions
    total_predictions = len(y_true)
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy


def _validate_regression_targets(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Validate input arrays for regression metrics.
    
    Performs validation checks to ensure the inputs are suitable
    for regression metric calculations.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like  
        Predicted target values
    
    Returns:
    --------
    y_true_validated : ndarray
        Validated true values array
    y_pred_validated : ndarray
        Validated predicted values array
    
    Raises:
    -------
    ValueError
        If inputs have incompatible shapes or contain invalid values
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Check dimensions
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-dimensional, got shape {y_true.shape}")
    
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred must be 1-dimensional, got shape {y_pred.shape}")
    
    # Check lengths match
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got y_true: {len(y_true)}, y_pred: {len(y_pred)}"
        )
    
    # Check for empty arrays
    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics for empty arrays")
    
    # Check for invalid values
    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains NaN or infinite values")
    
    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred contains NaN or infinite values")
    
    return y_true, y_pred


def _validate_classification_targets(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Validate input arrays for classification metrics.
    
    Performs validation checks to ensure the inputs are suitable
    for classification metric calculations.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    
    Returns:
    --------
    y_true_validated : ndarray
        Validated true labels array
    y_pred_validated : ndarray
        Validated predicted labels array
    
    Raises:
    -------
    ValueError
        If inputs have incompatible shapes or contain invalid values
    """
    # Convert to numpy arrays (keep original dtype for labels)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Check dimensions
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-dimensional, got shape {y_true.shape}")
    
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred must be 1-dimensional, got shape {y_pred.shape}")
    
    # Check lengths match
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got y_true: {len(y_true)}, y_pred: {len(y_pred)}"
        )
    
    # Check for empty arrays
    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics for empty arrays")
    
    return y_true, y_pred