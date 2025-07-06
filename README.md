# Customer Segmentation using K-Means Clustering

This project performs customer segmentation using the K-Means clustering algorithm. It groups customers based on their Age, Annual Income, and Spending Score to find marketing-friendly segments.

---

##  Dataset

The dataset contains 200 records with the following features:

- **CustomerID**
- **Gender**
- **Age**
- **Annual Income (k$)**
- **Spending Score (1–100)**

---

##  Steps Performed

### 1. Load and Standardize Data
- Selected numerical features: `Age`, `Annual Income`, `Spending Score`
- Standardized using `StandardScaler` to bring all features to the same scale

### 2. Apply K-Means Clustering
- Used K-Means from `sklearn.cluster`
- Initial clustering done with `k = 3`

### 3. Determine Optimal K (Elbow Method)
- Plotted Within-Cluster Sum of Squares (Inertia) vs Number of Clusters
- Identified optimal number of clusters: **K = 5**

### 4. Visualize Clusters (2D)
- Applied **PCA** to reduce the dataset from 3D to 2D
- Visualized clusters using a scatter plot with color-coding

### 5. Evaluate Clustering
- Calculated **Silhouette Score** to evaluate clustering quality

---

##  Tech Stack

- **Python**
- **Pandas** for data manipulation
- **Scikit-learn** for machine learning
- **Matplotlib** for visualization

---

##  Key Learnings

- How to preprocess and scale real-world data
- How K-Means works and how to choose K
- Dimensionality reduction with PCA
- Visualizing and evaluating unsupervised models

---

##  Output

- Clustered customers based on similarity in behavior
- Insights into distinct customer groups

---


