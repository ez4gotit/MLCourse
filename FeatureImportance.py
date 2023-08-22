import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate sample data for binary classification
X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_redundant=1, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

# Calculate feature importances
feature_importances = model.feature_importances_

# Plot feature importances
plt.bar(range(X.shape[1]), feature_importances)
plt.xticks(range(X.shape[1]), ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"])
plt.xlabel("Feature")
plt.ylabel("Feature Importance")
plt.title("Feature Importances in Random Forest")
plt.show()
