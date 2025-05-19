# Install necessary packages
!pip install -q scikit-learn pandas matplotlib seaborn

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="Target")
target_names = iris.target_names

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with PCA components and target labels
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['Target'] = y

# Display explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Visualize the PCA result
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Target', palette='Set1', s=100)
plt.title("PCA - Iris Dataset")
plt.xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)")
plt.ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)")
plt.legend(labels=target_names)
plt.grid(True)
plt.show()
