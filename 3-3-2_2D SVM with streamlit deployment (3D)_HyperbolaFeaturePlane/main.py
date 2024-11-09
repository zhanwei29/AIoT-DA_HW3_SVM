#run with command below: 
#streamlit run "file_path"
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

# Streamlit title and notation
st.title("3D SVM Decision Boundary Visualization with Hyperbolic Clusters")

# Add notation for users
st.markdown("""
**Note**: You can adjust parameters using either the sliders *or* the text input fields. 
These two methods work independently, meaning changes in one control method won't automatically update the other.
Choose one method (either sliders or text inputs) to control the parameters consistently.
""")

# Define default values
DEFAULT_SPREAD_X = 2.5
DEFAULT_SPREAD_Y = 2.5
DEFAULT_NOISE_LEVEL = 0.5

# Initialize session state if it doesn't exist
if 'spread_x' not in st.session_state:
    st.session_state.spread_x = DEFAULT_SPREAD_X
    st.session_state.spread_y = DEFAULT_SPREAD_Y
    st.session_state.noise_level = DEFAULT_NOISE_LEVEL

# Reset button to restore default values
if st.button("Reset"):
    st.session_state.spread_x = DEFAULT_SPREAD_X
    st.session_state.spread_y = DEFAULT_SPREAD_Y
    st.session_state.noise_level = DEFAULT_NOISE_LEVEL

# Sliders
spread_x_slider = st.slider("Spread along X-axis", 0.1, 10.0, float(st.session_state.spread_x), key="spread_x")
spread_y_slider = st.slider("Spread along Y-axis", 0.1, 10.0, float(st.session_state.spread_y), key="spread_y")
noise_level_slider = st.slider("Noise Level", 0.1, 5.0, float(st.session_state.noise_level), key="noise_level")

# Text inputs for precise control
def get_float_input(label, default_value, key):
    input_value = st.text_input(label, str(default_value), key=key)
    try:
        return float(input_value)
    except ValueError:
        st.warning(f"Please enter a valid number for {label}")
        return default_value

st.write("Alternatively, set values manually with these inputs:")
spread_x_input = get_float_input("Set Spread along X-axis", spread_x_slider, key="spread_x_input")
spread_y_input = get_float_input("Set Spread along Y-axis", spread_y_slider, key="spread_y_input")
noise_level_input = get_float_input("Set Noise Level", noise_level_slider, key="noise_level_input")

# Use values from text inputs if they differ from the sliders
spread_x = spread_x_input
spread_y = spread_y_input
noise_level = noise_level_input

# Generate hyperbolic clusters
np.random.seed(42)
N = 300  # Number of points per cluster

# Cluster 1 (upper branch of hyperbola)
x1 = np.random.uniform(-10, 10, N)
y1 = spread_y * np.sqrt(1 + (x1 ** 2) / spread_x**2) + np.random.normal(0, noise_level, N)  # y = sqrt(1 + x^2 / a^2) + noise
cluster1 = np.column_stack((x1, y1))

# Cluster 2 (lower branch of hyperbola)
x2 = np.random.uniform(-10, 10, N)
y2 = -spread_y * np.sqrt(1 + (x2 ** 2) / spread_x**2) + np.random.normal(0, noise_level, N)  # y = -sqrt(1 + x^2 / a^2) + noise
cluster2 = np.column_stack((x2, y2))

# Labels
Y1 = np.zeros(N)
Y2 = np.ones(N)

# Combine clusters and labels
X_2d = np.vstack((cluster1, cluster2))
Y = np.concatenate((Y1, Y2))

# Display 2D scatter plot
st.subheader("2D Scatter Plot of Hyperbolic Clusters with Labels Y=0 and Y=1")
fig, ax = plt.subplots()
ax.scatter(X_2d[Y == 0][:, 0], X_2d[Y == 0][:, 1], color="blue", label="Cluster 1")
ax.scatter(X_2d[Y == 1][:, 0], X_2d[Y == 1][:, 1], color="orange", label="Cluster 2")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.legend()
ax.set_title("2D Scatter Plot of Hyperbolic Clusters")
st.pyplot(fig)

# Step 3: Create X3 as a Gaussian function of X1 and X2
X3 = np.exp(- ((X_2d[:, 0]**2 / (2 * spread_x**2)) + (X_2d[:, 1]**2 / (2 * spread_y**2))) / noise_level)

# Normalize X3 values to a specified range, e.g., 0 to 1
X3 = X3 / np.max(X3)

# Combine X1, X2, and X3 for 3D plotting
X = np.column_stack((X_2d, X3))

# Display 3D scatter plot
st.subheader("3D Scatter Plot of Points with Gaussian Transformation for X3 (Hyperbolic Distribution)")
fig_3d = plt.figure(figsize=(10, 8))
ax = fig_3d.add_subplot(111, projection='3d')
ax.scatter(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], color="blue", marker="o", label="Y=0")
ax.scatter(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], color="red", marker="s", label="Y=1")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3 (Normalized)")
ax.legend()
ax.set_title("3D Scatter Plot with Gaussian Transformation for X3 (Hyperbolic Distribution)")
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
ax_hyperplane.set_title("3D Scatter Plot with Linear Separating Hyperplane (Hyperbolic Distribution)")
st.pyplot(fig_hyperplane)
