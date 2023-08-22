import numpy as np

def linear_regression_gradient_descent(X, y, learning_rate=0.01, num_iterations=1000):
    # Initialize parameters (slope and intercept)
    m, n = X.shape
    theta = np.zeros((n, 1))
    history = {'loss': [], 'theta': []}
    
    for iteration in range(num_iterations):
        # Calculate predictions
        predictions = np.dot(X, theta)
        
        # Calculate errors
        errors = predictions - y
        
        # Update parameters using gradient descent
        gradient = np.dot(X.T, errors) / m
        theta -= learning_rate * gradient
        
        # Calculate Mean Squared Error (MSE)
        mse = np.mean(errors**2)
        history['loss'].append(mse)
        history['theta'].append(theta.copy())
    
    return theta, history

# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 2

# Add bias term to X
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Perform linear regression using gradient descent
theta, history = linear_regression_gradient_descent(X_b, y)

# Calculate R-squared (R²)
y_pred = X_b.dot(theta)
sst = np.sum((y - np.mean(y))**2)
ssr = np.sum((y - y_pred)**2)
r2 = 1 - (ssr / sst)

print(f"Estimated coefficients (slope and intercept): {theta.ravel()}")
print(f"R-squared (R²): {r2:.2f}")
