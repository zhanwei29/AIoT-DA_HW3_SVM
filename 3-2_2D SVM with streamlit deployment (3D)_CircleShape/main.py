#run with command below: 
#streamlit run "d:/vs code/AIoT_Project/SupportVectorMachine/3-2_2D SVM with streamlit deployment (3D)_CircleShape/main.py"
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

# Streamlit title and notation
st.title("3D SVM Decision Boundary Visualization with Adjustable Gaussian Parameters")

# Add notation for users
st.markdown("""
**Note**: You can adjust parameters using either the sliders *or* the text input fields. 
These two methods work independently, meaning changes in one control method won't automatically update the other.
Choose one method (either sliders or text inputs) to control the parameters consistently.
""")

# Define default values
DEFAULT_DISTANCE_THRESHOLD = 4.25
DEFAULT_SEMI_MAJOR_AXIS = 3.0
DEFAULT_SEMI_MINOR_AXIS = 1.25
DEFAULT_SIGMA = 1.5

# Initialize session state if it doesn't exist
if 'distance_threshold' not in st.session_state:
    st.session_state.distance_threshold = DEFAULT_DISTANCE_THRESHOLD
    st.session_state.semi_major_axis = DEFAULT_SEMI_MAJOR_AXIS
    st.session_state.semi_minor_axis = DEFAULT_SEMI_MINOR_AXIS
    st.session_state.sigma = DEFAULT_SIGMA

# Reset button to restore default values
if st.button("Reset"):
    st.session_state.distance_threshold = DEFAULT_DISTANCE_THRESHOLD
    st.session_state.semi_major_axis = DEFAULT_SEMI_MAJOR_AXIS
    st.session_state.semi_minor_axis = DEFAULT_SEMI_MINOR_AXIS
    st.session_state.sigma = DEFAULT_SIGMA

# Adjustable parameters using sliders
distance_threshold = st.slider("Distance Threshold", 0.1, 10.0, st.session_state.distance_threshold, key="distance_threshold")
semi_major_axis = st.slider("Semi-Major Axis for Gaussian Function", 0.1, 10.0, st.session_state.semi_major_axis, key="semi_major_axis")
semi_minor_axis = st.slider("Semi-Minor Axis for Gaussian Function", 0.1, 10.0, st.session_state.semi_minor_axis, key="semi_minor_axis")
sigma = st.slider("Sigma for Gaussian Spread", 1.0, 20.0, st.session_state.sigma, key="sigma")

# The rest of the code for data generation, plotting, etc., follows here...
# Step 1: Generate first cluster centered at (0,0) with variance 10
np.random.seed(42)
C1 = [0, 0]
covariance = [[10, 0], [0, 10]]
X1_cluster1 = np.random.multivariate_normal(C1, covariance, 300)
distances_C1 = np.sqrt((X1_cluster1[:, 0])**2 + (X1_cluster1[:, 1])**2)
Y1 = np.where(distances_C1 < distance_threshold, 0, 1)

# Step 2: Generate second cluster centered at (10,10) with variance 10
C2 = [10, 10]
X2_cluster2 = np.random.multivariate_normal(C2, covariance, 300)
distances_C2 = np.sqrt((X2_cluster2[:, 0] - 10)**2 + (X2_cluster2[:, 1] - 10)**2)
Y2 = np.where(distances_C2 < distance_threshold, 0, 1)

# Combine clusters
X_2d = np.vstack((X1_cluster1, X2_cluster2))
Y = np.concatenate((Y1, Y2))

# Display 2D scatter plot
st.subheader("2D Scatter Plot of Two Gaussian Clusters with Labels Y=0 and Y=1")
fig, ax = plt.subplots()
ax.scatter(X_2d[Y == 0][:, 0], X_2d[Y == 0][:, 1], color="blue", label="Y=0")
ax.scatter(X_2d[Y == 1][:, 0], X_2d[Y == 1][:, 1], color="orange", label="Y=1")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.legend()
ax.set_title("2D Scatter Plot of Two Clusters")
st.pyplot(fig)

# Step 3: Create X3 as a Gaussian function of X1 and X2
X3 = np.exp(- ((X_2d[:, 0]**2 / (2 * semi_major_axis**2)) + (X_2d[:, 1]**2 / (2 * semi_minor_axis**2))) / sigma)

# Normalize X3 values to a specified range, e.g., 0 to 1
X3 = X3 / np.max(X3)

# Combine X1, X2, and X3 for 3D plotting
X = np.column_stack((X_2d, X3))

# Display 3D scatter plot
st.subheader("3D Scatter Plot of Points with Gaussian Transformation for X3")
fig_3d = plt.figure(figsize=(10, 8))
ax = fig_3d.add_subplot(111, projection='3d')
ax.scatter(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], color="blue", marker="o", label="Y=0")
ax.scatter(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], color="red", marker="s", label="Y=1")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3 (Normalized)")
ax.legend()
ax.set_title("3D Scatter Plot with Gaussian Transformation for X3")
st.pyplot(fig_3d)

# Step 4: Train an SVM model to find a linear separating hyperplane in 3D
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X, Y)

# Meshgrid for hyperplane
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

# Calculate hyperplane Z values
coef = svm_model.coef_[0]
intercept = svm_model.intercept_[0]
zz = -(coef[0] * xx + coef[1] * yy + intercept) / coef[2]

# Plot the hyperplane
fig_hyperplane = plt.figure(figsize=(10, 8))
ax_hyperplane = fig_hyperplane.add_subplot(111, projection='3d')
ax_hyperplane.scatter(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], color="blue", marker="o", label="Y=0")
ax_hyperplane.scatter(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], color="red", marker="s", label="Y=1")
ax_hyperplane.plot_surface(xx, yy, zz, color="lightblue", alpha=0.5)
ax_hyperplane.set_xlabel("X1")
ax_hyperplane.set_ylabel("X2")
ax_hyperplane.set_zlabel("X3 (Normalized)")
ax_hyperplane.legend()
ax_hyperplane.set_title("3D Scatter Plot with Linear Separating Hyperplane")
st.pyplot(fig_hyperplane)
