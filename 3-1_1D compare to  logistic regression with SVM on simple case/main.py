import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# CRISP-DM Steps
# 1. Business Understanding
# Objective: Compare logistic regression and SVM (with RBF kernel) on a simple 1D classification problem

# 2. Data Understanding
# Generate data based on defined rules and constraints

# Generate random variables X(i) with values between 0 and 1000
np.random.seed(42)  # For reproducibility
X = np.random.uniform(0, 1000, 300)

# Determine Y(i) based on rules: Y(i) = 1 if 500 < X(i) < 800; Y(i) = 0, otherwise
Y = np.where((X > 500) & (X < 800), 1, 0)

# Reshape X for compatibility with scikit-learn models
X_reshaped = X.reshape(-1, 1)

# 3. Data Preparation
# No additional data cleaning or preprocessing is needed for this simple dataset

# 4. Modeling
# Logistic Regression Model
log_reg = LogisticRegression().fit(X_reshaped, Y)
Y1 = log_reg.predict(X_reshaped)

# SVM Model with RBF Kernel
svm_rbf = SVC(kernel="rbf", C=1.0).fit(X_reshaped, Y)
Y2_rbf = svm_rbf.predict(X_reshaped)

# 5. Evaluation
# Plotting results for comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot for Logistic Regression
ax1.scatter(X, Y, color="blue", label="Actual Y")
ax1.scatter(X, Y1, color="red", alpha=0.5, label="Logistic Regression Prediction Y1")
ax1.axvline(x=500, color="black", linestyle="--", label="Decision Boundary")
ax1.axvline(x=800, color="black", linestyle="--")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()
ax1.set_title("Logistic Regression Results")

# Plot for SVM with RBF Kernel
ax2.scatter(X, Y, color="blue", label="Actual Y")
ax2.scatter(X, Y2_rbf, color="purple", alpha=0.5, label="SVM (RBF) Prediction Y2")
ax2.axvline(x=500, color="black", linestyle="--", label="Decision Boundary")
ax2.axvline(x=800, color="black", linestyle="--")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.legend()
ax2.set_title("SVM with RBF Kernel Results")

plt.tight_layout()
plt.show()
