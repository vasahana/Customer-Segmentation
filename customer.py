import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("=== Simplified Customer Segmentation ===\n")

# 1. Generate sample data
print("1. Generating sample customer data...")
np.random.seed(42)
n_samples = 300
data = {
    'CustomerID': range(1, n_samples + 1),
    'Age': np.random.randint(18, 70, n_samples),
    'Annual_Income': np.random.normal(50000, 15000, n_samples).astype(int),
    'Spending_Score': np.random.randint(1, 100, n_samples),
    'Purchase_Frequency': np.random.poisson(8, n_samples),
}
df = pd.DataFrame(data)

# Create some realistic patterns
df['Spending_Score'] = (df['Annual_Income'] / 1000 + 
                        df['Purchase_Frequency'] * 3 + 
                        np.random.normal(0, 15, n_samples)).astype(int)
df['Spending_Score'] = np.clip(df['Spending_Score'], 1, 100)

print(f"Data shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData description:")
print(df.describe())

# 2. Select features for clustering
features = ['Annual_Income', 'Spending_Score', 'Age', 'Purchase_Frequency']
X = df[features]
print(f"\n2. Features for clustering: {features}")

# 3. Scale the features
print("3. Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Find optimal number of clusters using elbow method
print("4. Finding optimal clusters...")
wcss = [] # Within-cluster sum of squares
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.grid(True, alpha=0.3)
plt.show()

# 5. Perform K-means clustering with optimal k
optimal_k = 4 # Based on elbow curve
print(f"\n5. Performing K-means clustering with {optimal_k} clusters...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters
print("Clustering completed!")
print(f"Cluster distribution:\n{df['Cluster'].value_counts().sort_index()}")

# 6. Visualize the clusters
print("\n6. Creating visualizations...")
# Visualization 1: Income vs Spending Score
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['Annual_Income'], df['Spending_Score'], 
                      c=df['Cluster'], cmap='viridis', alpha=0.7, s=60)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segments: Income vs Spending')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# Visualization 2: PCA for better cluster visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['PC1'], df['PC2'], c=df['Cluster'], 
                      cmap='viridis', alpha=0.7, s=60)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Customer Segments - PCA Visualization')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# 7. Analyze cluster characteristics
print("\n7. Analyzing cluster characteristics...")
cluster_analysis = df.groupby('Cluster').agg({
    'Annual_Income': ['mean', 'std', 'min', 'max'],
    'Spending_Score': ['mean', 'std', 'min', 'max'],
    'Age': ['mean', 'std'],
    'Purchase_Frequency': ['mean', 'std'],
    'CustomerID': 'count'
}).round(2)

cluster_analysis.columns = ['_'.join(col).strip() for col in cluster_analysis.columns.values]
cluster_analysis = cluster_analysis.rename(columns={'CustomerID_count': 'Customer_Count'})
print("Cluster Analysis:")
print(cluster_analysis)

# 8. Create cluster profiles
print("\n8. Cluster Profiles:")
for cluster in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster]
    
    print(f"\n--- Cluster {cluster} ({len(cluster_data)} customers) ---")
    print(f"Average Income: ${cluster_data['Annual_Income'].mean():.0f}")
    print(f"Average Spending Score: {cluster_data['Spending_Score'].mean():.1f}")
    print(f"Average Age: {cluster_data['Age'].mean():.1f}")
    print(f"Average Purchase Frequency: {cluster_data['Purchase_Frequency'].mean():.1f}")
    
    # Create segment description
    income_level = "High" if cluster_data['Annual_Income'].mean() > 55000 else "Medium" if cluster_data['Annual_Income'].mean() > 45000 else "Low"
    spending_level = "High" if cluster_data['Spending_Score'].mean() > 60 else "Medium" if cluster_data['Spending_Score'].mean() > 40 else "Low"
    
    print(f"Segment Type: {income_level} Income, {spending_level} Spenders")

# 9. Additional visualization: Box plots
print("\n9. Creating box plots for cluster comparison...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
features_to_plot = ['Annual_Income', 'Spending_Score', 'Age', 'Purchase_Frequency']
for i, feature in enumerate(features_to_plot):
    row, col = i // 2, i % 2
    df.boxplot(column=feature, by='Cluster', ax=axes[row, col])
    axes[row, col].set_title(f'{feature} by Cluster')

plt.suptitle('Cluster Characteristics Comparison')
plt.tight_layout()
plt.show()

# 10. Correlation heatmap
print("\n10. Feature correlation heatmap...")
plt.figure(figsize=(10, 8))
correlation_matrix = df[features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

print("\n=== ANALYSIS COMPLETE ===")
print(f"Total customers analyzed: {len(df)}")
print(f"Number of segments identified: {optimal_k}")
print(f"\nSegment sizes:")
for cluster in sorted(df['Cluster'].unique()):
    count = len(df[df['Cluster'] == cluster])
    percentage = (count / len(df)) * 100
    print(f"Cluster {cluster}: {count} customers ({percentage:.1f}%)")