import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate example biased data
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                           weights=[0.7, 0.3], random_state=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Print the number of data points in each class
print(f"Class 0 data points: {len(y_train[y_train == 0])}")
print(f"Class 1 data points: {len(y_train[y_train == 1])}")

# Train a logistic regression model on the biased data
model_biased = LogisticRegression()
model_biased.fit(X_train, y_train)

# Plot the biased data
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Biased Data')
plt.show()

# Evaluate accuracy on biased test data
y_test_pred_biased = model_biased.predict(X_test)
accuracy_biased = accuracy_score(y_test, y_test_pred_biased)
print(f"Accuracy on biased test data: {accuracy_biased:.2f}")

# Mitigate bias using oversampling and undersampling
X_resampled = np.vstack([X_train[y_train == 1], X_train[y_train == 0][:150]])
y_resampled = np.hstack([y_train[y_train == 1], y_train[y_train == 0][:150]])

# Print the number of data points in each class after mitigation
print(f"Class 0 data points after mitigation: {len(y_resampled[y_resampled == 0])}")
print(f"Class 1 data points after mitigation: {len(y_resampled[y_resampled == 1])}")

# Train a logistic regression model on the mitigated data
model_mitigated = LogisticRegression()
model_mitigated.fit(X_resampled, y_resampled)

# Plot the mitigated data and decision boundary
plt.scatter(X_resampled[y_resampled == 0][:, 0], X_resampled[y_resampled == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X_resampled[y_resampled == 1][:, 0], X_resampled[y_resampled == 1][:, 1], color='red', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100),
                     np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100))
Z = model_mitigated.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)

plt.legend()
plt.title('Mitigated Data')
plt.show()

# Evaluate accuracy on mitigated test data
y_test_pred_mitigated = model_mitigated.predict(X_test)
accuracy_mitigated = accuracy_score(y_test, y_test_pred_mitigated)
print(f"Accuracy on mitigated test data: {accuracy_mitigated:.2f}")
