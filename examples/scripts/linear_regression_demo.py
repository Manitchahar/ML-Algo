"""
Linear Regression Example Script

This script demonstrates how to use the Linear Regression implementation
with detailed explanations and comments for educational purposes.

The example includes:
- Data generation and visualization
- Model training with both normal equation and gradient descent
- Performance evaluation and comparison
- Visualization of results

Author: ML-Algo Contributors
Date: 2025
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.supervised.regression import LinearRegression
from utils.metrics import mean_squared_error, r2_score, root_mean_squared_error


def generate_sample_data(n_samples: int = 100, noise_level: float = 0.1, seed: int = 42):
    """
    Generate synthetic linear regression data for demonstration.
    
    Creates a dataset where the target variable is a linear combination
    of the input features plus some random noise.
    
    Mathematical Relationship:
    y = 3*x₁ + 2*x₂ + 1 + ε
    
    Where:
    - x₁, x₂: input features (random normal distribution)
    - ε: Gaussian noise ~ N(0, noise_level²)
    
    Parameters:
    -----------
    n_samples : int, default=100
        Number of samples to generate
    noise_level : float, default=0.1
        Standard deviation of Gaussian noise added to targets
    seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X : ndarray, shape (n_samples, 2)
        Feature matrix with 2 features
    y : ndarray, shape (n_samples,)
        Target values following linear relationship
    true_coefficients : ndarray, shape (2,)
        True coefficients used to generate data [3, 2]
    true_intercept : float
        True intercept used to generate data (1.0)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate random features from standard normal distribution
    # Shape: (n_samples, 2)
    X = np.random.randn(n_samples, 2)
    
    # Define true linear relationship parameters
    true_coefficients = np.array([3.0, 2.0])  # Coefficients for x₁ and x₂
    true_intercept = 1.0                       # Intercept term
    
    # Generate target values using linear relationship
    # y = X @ true_coefficients + true_intercept + noise
    y_clean = X @ true_coefficients + true_intercept
    
    # Add Gaussian noise to make the problem realistic
    noise = np.random.normal(0, noise_level, n_samples)
    y = y_clean + noise
    
    return X, y, true_coefficients, true_intercept


def train_and_evaluate_model(X_train, y_train, X_test, y_test, method='normal_equation'):
    """
    Train a Linear Regression model and evaluate its performance.
    
    This function demonstrates the complete machine learning workflow:
    1. Model initialization with specified method
    2. Training on training data
    3. Prediction on test data
    4. Performance evaluation using multiple metrics
    
    Parameters:
    -----------
    X_train : ndarray, shape (n_train_samples, n_features)
        Training feature matrix
    y_train : ndarray, shape (n_train_samples,)
        Training target values
    X_test : ndarray, shape (n_test_samples, n_features)
        Test feature matrix
    y_test : ndarray, shape (n_test_samples,)
        Test target values
    method : str, default='normal_equation'
        Training method to use ('normal_equation' or 'gradient_descent')
        
    Returns:
    --------
    model : LinearRegression
        Trained Linear Regression model
    train_predictions : ndarray
        Predictions on training data
    test_predictions : ndarray
        Predictions on test data
    metrics : dict
        Dictionary containing evaluation metrics
    """
    print(f"\n=== Training Linear Regression using {method.replace('_', ' ').title()} ===")
    
    # Initialize model with appropriate parameters
    if method == 'normal_equation':
        model = LinearRegression(method='normal_equation')
    else:  # gradient_descent
        model = LinearRegression(
            method='gradient_descent',
            learning_rate=0.01,      # Learning rate - controls step size
            max_iterations=1000,     # Maximum iterations to prevent infinite loops
            tolerance=1e-6           # Convergence tolerance
        )
    
    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Make predictions on both training and test sets
    print("Making predictions...")
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Calculate evaluation metrics
    print("Calculating evaluation metrics...")
    
    # Training metrics
    train_mse = mean_squared_error(y_train, train_predictions)
    train_rmse = root_mean_squared_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    
    # Test metrics
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = root_mean_squared_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    # Store metrics in dictionary
    metrics = {
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }
    
    # Display model parameters
    print(f"\nModel Parameters:")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    # Display performance metrics
    print(f"\nTraining Performance:")
    print(f"  MSE:  {train_mse:.6f}")
    print(f"  RMSE: {train_rmse:.6f}")
    print(f"  R²:   {train_r2:.6f}")
    
    print(f"\nTest Performance:")
    print(f"  MSE:  {test_mse:.6f}")
    print(f"  RMSE: {test_rmse:.6f}")
    print(f"  R²:   {test_r2:.6f}")
    
    # Additional information for gradient descent
    if method == 'gradient_descent':
        print(f"\nGradient Descent Information:")
        print(f"  Iterations used: {model.n_iterations_}")
        print(f"  Final cost: {model.cost_history_[-1]:.6f}")
    
    return model, train_predictions, test_predictions, metrics


def plot_results(X_test, y_test, test_predictions, model, method):
    """
    Create visualizations to show model performance.
    
    This function creates several plots to help understand the model:
    1. Predicted vs Actual values scatter plot
    2. Residuals plot to check for patterns
    3. Cost function history (for gradient descent only)
    
    Parameters:
    -----------
    X_test : ndarray
        Test feature matrix
    y_test : ndarray
        True test target values
    test_predictions : ndarray
        Predicted test target values
    model : LinearRegression
        Trained model
    method : str
        Training method used
    """
    # Create subplots
    if method == 'gradient_descent':
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Predicted vs Actual
    axes[0].scatter(y_test, test_predictions, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
    
    # Add perfect prediction line (diagonal)
    min_val = min(y_test.min(), test_predictions.min())
    max_val = max(y_test.max(), test_predictions.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[0].set_xlabel('True Values', fontsize=12)
    axes[0].set_ylabel('Predicted Values', fontsize=12)
    axes[0].set_title(f'Predicted vs Actual Values\n({method.replace("_", " ").title()})', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add R² score to the plot
    r2 = r2_score(y_test, test_predictions)
    axes[0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=11, verticalalignment='top')
    
    # Plot 2: Residuals
    residuals = y_test - test_predictions
    axes[1].scatter(test_predictions, residuals, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Values', fontsize=12)
    axes[1].set_ylabel('Residuals (True - Predicted)', fontsize=12)
    axes[1].set_title('Residuals Plot', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Add residual statistics
    residual_std = np.std(residuals)
    axes[1].text(0.05, 0.95, f'Residual Std: {residual_std:.4f}', 
                transform=axes[1].transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=11, verticalalignment='top')
    
    # Plot 3: Cost history (only for gradient descent)
    if method == 'gradient_descent':
        axes[2].plot(model.cost_history_, 'b-', linewidth=2)
        axes[2].set_xlabel('Iteration', fontsize=12)
        axes[2].set_ylabel('Cost (MSE)', fontsize=12)
        axes[2].set_title('Cost Function During Training', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        
        # Add convergence information
        final_cost = model.cost_history_[-1]
        axes[2].text(0.05, 0.95, f'Final Cost: {final_cost:.6f}\nIterations: {model.n_iterations_}', 
                    transform=axes[2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=11, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()


def compare_methods(X_train, y_train, X_test, y_test, true_coefficients, true_intercept):
    """
    Compare Normal Equation and Gradient Descent methods.
    
    This function trains models using both methods and compares:
    - Training time
    - Final parameters
    - Prediction accuracy
    - Convergence behavior
    
    Parameters:
    -----------
    X_train, y_train : ndarray
        Training data
    X_test, y_test : ndarray
        Test data
    true_coefficients : ndarray
        True coefficients used to generate data
    true_intercept : float
        True intercept used to generate data
    """
    print("\n" + "="*60)
    print("COMPARISON: Normal Equation vs Gradient Descent")
    print("="*60)
    
    methods = ['normal_equation', 'gradient_descent']
    results = {}
    
    # Train models with both methods
    for method in methods:
        model, train_pred, test_pred, metrics = train_and_evaluate_model(
            X_train, y_train, X_test, y_test, method
        )
        
        # Store results
        results[method] = {
            'model': model,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'metrics': metrics
        }
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nTrue Parameters:")
    print(f"  Coefficients: {true_coefficients}")
    print(f"  Intercept: {true_intercept:.4f}")
    
    print(f"\n{'Method':<20} {'Test R²':<10} {'Test RMSE':<12} {'Coef Error':<12} {'Intercept Error':<15}")
    print("-" * 75)
    
    for method in methods:
        model = results[method]['model']
        metrics = results[method]['metrics']
        
        # Calculate parameter errors
        coef_error = np.linalg.norm(model.coef_ - true_coefficients)
        intercept_error = abs(model.intercept_ - true_intercept)
        
        print(f"{method:<20} {metrics['test_r2']:<10.6f} {metrics['test_rmse']:<12.6f} "
              f"{coef_error:<12.6f} {intercept_error:<15.6f}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, method in enumerate(methods):
        test_pred = results[method]['test_predictions']
        
        axes[i].scatter(y_test, test_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), test_pred.min())
        max_val = max(y_test.max(), test_pred.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        axes[i].set_xlabel('True Values', fontsize=12)
        axes[i].set_ylabel('Predicted Values', fontsize=12)
        axes[i].set_title(f'{method.replace("_", " ").title()}', fontsize=14)
        axes[i].grid(True, alpha=0.3)
        
        # Add R² score
        r2 = results[method]['metrics']['test_r2']
        axes[i].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[i].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=11, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    return results


def main():
    """
    Main function that demonstrates the complete Linear Regression workflow.
    
    This function orchestrates the entire demonstration:
    1. Data generation
    2. Train/test split
    3. Model training and evaluation
    4. Results visualization
    5. Method comparison
    """
    print("=" * 60)
    print("LINEAR REGRESSION DEMONSTRATION")
    print("=" * 60)
    print("\nThis example demonstrates:")
    print("- Synthetic data generation")
    print("- Model training with Normal Equation and Gradient Descent")
    print("- Performance evaluation using multiple metrics")
    print("- Results visualization and comparison")
    
    # Step 1: Generate synthetic data
    print(f"\n{'-'*40}")
    print("STEP 1: DATA GENERATION")
    print(f"{'-'*40}")
    
    n_samples = 200
    noise_level = 0.2
    print(f"Generating {n_samples} samples with noise level {noise_level}")
    
    X, y, true_coefficients, true_intercept = generate_sample_data(
        n_samples=n_samples,
        noise_level=noise_level,
        seed=42
    )
    
    print(f"Data shape: X {X.shape}, y {y.shape}")
    print(f"True relationship: y = {true_coefficients[0]:.1f}*x1 + {true_coefficients[1]:.1f}*x2 + {true_intercept:.1f} + noise")
    
    # Step 2: Split data into training and test sets
    print(f"\n{'-'*40}")
    print("STEP 2: DATA SPLITTING")
    print(f"{'-'*40}")
    
    # Simple train/test split (80/20)
    split_idx = int(0.8 * n_samples)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 3: Train and evaluate models
    print(f"\n{'-'*40}")
    print("STEP 3: MODEL TRAINING & EVALUATION")
    print(f"{'-'*40}")
    
    # Train with Normal Equation
    model_ne, train_pred_ne, test_pred_ne, metrics_ne = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, method='normal_equation'
    )
    
    # Train with Gradient Descent
    model_gd, train_pred_gd, test_pred_gd, metrics_gd = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, method='gradient_descent'
    )
    
    # Step 4: Visualize results
    print(f"\n{'-'*40}")
    print("STEP 4: RESULTS VISUALIZATION")
    print(f"{'-'*40}")
    
    print("Generating plots for Normal Equation method...")
    plot_results(X_test, y_test, test_pred_ne, model_ne, 'normal_equation')
    
    print("Generating plots for Gradient Descent method...")
    plot_results(X_test, y_test, test_pred_gd, model_gd, 'gradient_descent')
    
    # Step 5: Compare methods
    print(f"\n{'-'*40}")
    print("STEP 5: METHOD COMPARISON")
    print(f"{'-'*40}")
    
    comparison_results = compare_methods(
        X_train, y_train, X_test, y_test, true_coefficients, true_intercept
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    print("\nKey Takeaways:")
    print("1. Both methods produce very similar results for this problem")
    print("2. Normal Equation is faster but requires matrix inversion")
    print("3. Gradient Descent is more flexible and works with large datasets")
    print("4. The cost function visualization shows convergence behavior")
    print("5. Residuals plot helps identify potential model issues")
    
    print(f"\nBoth models successfully recovered the true parameters:")
    print(f"True coefficients: {true_coefficients}")
    print(f"Normal Equation:   {model_ne.coef_}")
    print(f"Gradient Descent:  {model_gd.coef_}")


if __name__ == "__main__":
    # Run the demonstration
    main()