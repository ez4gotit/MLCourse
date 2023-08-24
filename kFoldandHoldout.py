import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate example data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, size=len(X))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Plot the data
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Testing Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Data Distribution')
plt.show()

# Sequentially plot polynomial fits with increasing degrees
degrees = [1, 2, 5, 10, 15]
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Testing Data')

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    y_train_pred_kfold = cross_val_predict(LinearRegression(), X_train_poly, y_train, cv=kf)
    y_test_pred_kfold = LinearRegression().fit(X_train_poly, y_train).predict(X_test_poly)

    mse_train_kfold = mean_squared_error(y_train, y_train_pred_kfold)
    mse_test_kfold = mean_squared_error(y_test, y_test_pred_kfold)

    # Holdout Validation
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_train_pred_holdout = model.predict(X_train_poly)
    y_test_pred_holdout = model.predict(X_test_poly)

    mse_train_holdout = mean_squared_error(y_train, y_train_pred_holdout)
    mse_test_holdout = mean_squared_error(y_test, y_test_pred_holdout)

    x_range = np.linspace(0, 5, 100).reshape(-1, 1)
    x_range_poly = poly.transform(x_range)
    y_range_pred = model.predict(x_range_poly)

    plt.plot(x_range, y_range_pred,
             label=f'Degree {degree} (KFold MSE: {mse_train_kfold:.2f}/{mse_test_kfold:.2f}, Holdout MSE: {mse_train_holdout:.2f}/{mse_test_holdout:.2f})')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Sequential Polynomial Regression Fits with K-Fold and Holdout Validation')
plt.show()
