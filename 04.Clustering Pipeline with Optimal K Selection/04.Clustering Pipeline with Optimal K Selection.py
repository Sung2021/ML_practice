import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, adjusted_rand_score, silhouette_samples
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
np.random.seed(42)

# =============================================================================
# STEP 1: Generate Realistic scRNA-seq Style Cancer Data
# =============================================================================
print("=" * 70)
print("STEP 1: Realistic scRNA-seq Cancer Data Generation")
print("=" * 70)

# Simulate 3 cancer subtypes with realistic characteristics
n_samples_per_subtype = [180, 120, 100]  # Imbalanced cluster sizes
n_samples = sum(n_samples_per_subtype)
n_genes = 2000  # Highly variable genes
n_patients = 6  # Multiple patients for batch effect

print(f"Simulating cancer subtypes:")
print(f"  - Subtype 1 (Proliferative): {n_samples_per_subtype[0]} cells")
print(f"  - Subtype 2 (Invasive): {n_samples_per_subtype[1]} cells")
print(f"  - Subtype 3 (Inflammatory): {n_samples_per_subtype[2]} cells")
print(f"Total cells: {n_samples}")
print(f"Number of genes: {n_genes}")
print(f"Number of patients: {n_patients}")

# Subtype 1: Proliferative state (high cell cycle genes)
X1_base = np.random.negative_binomial(n=5, p=0.3, size=(n_samples_per_subtype[0], n_genes))
X1_signature = np.zeros((n_samples_per_subtype[0], n_genes))
X1_signature[:, :200] = np.random.negative_binomial(n=10, p=0.2, size=(n_samples_per_subtype[0], 200))
X1 = X1_base + X1_signature

# Subtype 2: Invasive state (high EMT genes)
X2_base = np.random.negative_binomial(n=5, p=0.3, size=(n_samples_per_subtype[1], n_genes))
X2_signature = np.zeros((n_samples_per_subtype[1], n_genes))
X2_signature[:, 200:400] = np.random.negative_binomial(n=12, p=0.15, size=(n_samples_per_subtype[1], 200))
X2 = X2_base + X2_signature

# Subtype 3: Inflammatory state (high immune response genes)
X3_base = np.random.negative_binomial(n=5, p=0.3, size=(n_samples_per_subtype[2], n_genes))
X3_signature = np.zeros((n_samples_per_subtype[2], n_genes))
X3_signature[:, 400:600] = np.random.negative_binomial(n=15, p=0.1, size=(n_samples_per_subtype[2], 200))
X3 = X3_base + X3_signature

X = np.vstack([X1, X2, X3]).astype(float)

# Add patient-specific batch effects
patient_ids = np.repeat(np.arange(n_patients), n_samples // n_patients)
for patient in range(n_patients):
    patient_mask = patient_ids == patient
    batch_effect = np.random.normal(0, 0.3, n_genes)
    X[patient_mask] += batch_effect

# Add dropout events (scRNA-seq characteristic)
dropout_rate = 0.15
dropout_mask = np.random.random(X.shape) < dropout_rate
X[dropout_mask] = 0

# Add missing data
missing_rate = 0.05
missing_mask = np.random.random(X.shape) < missing_rate
X[missing_mask] = np.nan

# Impute missing values with gene mean
col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])

# Log transformation (common in scRNA-seq)
X = np.log1p(X)

# Create labels
true_labels = np.array([0]*n_samples_per_subtype[0] + 
                       [1]*n_samples_per_subtype[1] + 
                       [2]*n_samples_per_subtype[2])

# Create dataframe
gene_names = [f"Gene_{i+1}" for i in range(n_genes)]
df = pd.DataFrame(X, columns=gene_names)
df["True_Subtype"] = true_labels
df["Patient_ID"] = patient_ids

subtype_names = {0: "Proliferative", 1: "Invasive", 2: "Inflammatory"}
df["Subtype_Name"] = df["True_Subtype"].map(subtype_names)

print(f"\nData characteristics:")
print(f"  - Zero expression rate: {(X == 0).sum() / X.size * 100:.1f}%")
print(f"  - Mean expression: {X.mean():.3f}")
print(f"  - Std expression: {X.std():.3f}")

print(f"\nSubtype distribution:")
print(df["Subtype_Name"].value_counts())

print(f"\nPatient distribution:")
print(df.groupby(["Patient_ID", "Subtype_Name"]).size().unstack(fill_value=0))

# =============================================================================
# STEP 2: Data Scaling
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Data Scaling")
print("=" * 70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Scaling completed")
print(f"Mean of scaled data: {X_scaled.mean():.6f}")
print(f"Std of scaled data: {X_scaled.std():.6f}")

# =============================================================================
# STEP 3: Determine Optimal Number of Clusters
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Optimal Cluster Number Selection")
print("=" * 70)

k_range = range(2, 11)
inertias = []
silhouettes = []
bics = []

print("Testing cluster numbers from 2 to 10...")
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    
    inertias.append(kmeans_temp.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels_temp))
    
    gmm_temp = GaussianMixture(n_components=k, random_state=42)
    gmm_temp.fit(X_scaled)
    bics.append(gmm_temp.bic(X_scaled))

optimal_k_df = pd.DataFrame({
    "K": list(k_range),
    "Inertia": inertias,
    "Silhouette": silhouettes,
    "BIC": bics
})

print("\nOptimal K Selection Results:")
print(optimal_k_df.to_string(index=False))

optimal_k_silhouette = optimal_k_df.loc[optimal_k_df["Silhouette"].idxmax(), "K"]
optimal_k_bic = optimal_k_df.loc[optimal_k_df["BIC"].idxmin(), "K"]

print(f"\nRecommended K by Silhouette Score: {optimal_k_silhouette}")
print(f"Recommended K by BIC: {optimal_k_bic}")

# Visualization of optimal K
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(k_range, inertias, marker='o', linewidth=2)
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (Within-cluster Sum of Squares)")
axes[0].set_title("Elbow Method")
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouettes, marker='o', linewidth=2, color='orange')
axes[1].axvline(optimal_k_silhouette, color='red', linestyle='--', alpha=0.7, 
                label=f'Optimal K={optimal_k_silhouette}')
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Analysis")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(k_range, bics, marker='o', linewidth=2, color='green')
axes[2].axvline(optimal_k_bic, color='red', linestyle='--', alpha=0.7, 
                label=f'Optimal K={optimal_k_bic}')
axes[2].set_xlabel("Number of Clusters (K)")
axes[2].set_ylabel("BIC (Lower is Better)")
axes[2].set_title("Bayesian Information Criterion")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# STEP 4: Silhouette Plot for Optimal K
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Detailed Silhouette Analysis for Optimal K")
print("=" * 70)

optimal_k = int(optimal_k_silhouette)
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
optimal_labels = kmeans_optimal.fit_predict(X_scaled)

silhouette_vals = silhouette_samples(X_scaled, optimal_labels)
silhouette_avg = silhouette_score(X_scaled, optimal_labels)

print(f"Average Silhouette Score for K={optimal_k}: {silhouette_avg:.4f}")

# Silhouette plot
fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 10

for i in range(optimal_k):
    cluster_silhouette_vals = silhouette_vals[optimal_labels == i]
    cluster_silhouette_vals.sort()
    
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = plt.cm.tab10(i / optimal_k)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                     facecolor=color, edgecolor=color, alpha=0.7)
    
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2, 
           label=f'Average: {silhouette_avg:.3f}')
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster Label")
ax.set_title(f"Silhouette Plot for K={optimal_k}")
ax.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# STEP 5: Apply Three Clustering Methods with Optimal K
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Apply Clustering Methods")
print("=" * 70)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)

print(f"\nKMeans (K={optimal_k}):")
print(f"  Silhouette Score: {kmeans_silhouette:.4f}")
print(f"  ARI vs True Labels: {kmeans_ari:.4f}")

gmm = GaussianMixture(n_components=optimal_k, covariance_type="full", random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_silhouette = silhouette_score(X_scaled, gmm_labels)
gmm_ari = adjusted_rand_score(true_labels, gmm_labels)

print(f"\nGaussian Mixture Model (K={optimal_k}):")
print(f"  Silhouette Score: {gmm_silhouette:.4f}")
print(f"  ARI vs True Labels: {gmm_ari:.4f}")

linkage_matrix = linkage(X_scaled, method="ward")
hierarchical_labels = fcluster(linkage_matrix, t=optimal_k, criterion="maxclust") - 1
hier_silhouette = silhouette_score(X_scaled, hierarchical_labels)
hier_ari = adjusted_rand_score(true_labels, hierarchical_labels)

print(f"\nHierarchical Clustering (K={optimal_k}):")
print(f"  Silhouette Score: {hier_silhouette:.4f}")
print(f"  ARI vs True Labels: {hier_ari:.4f}")

# =============================================================================
# STEP 6: PCA Visualization
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: PCA Visualization")
print("=" * 70)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance by PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"Explained variance by PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")

df_vis = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "True": df["Subtype_Name"],
    "KMeans": kmeans_labels,
    "GMM": gmm_labels,
    "Hierarchical": hierarchical_labels,
    "Patient": patient_ids
})

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: Clustering results
for i, (method, labels) in enumerate([("True Subtypes", "True"), 
                                       ("KMeans", "KMeans"), 
                                       ("GMM", "GMM")]):
    if method == "True Subtypes":
        sns.scatterplot(x="PC1", y="PC2", hue=labels, data=df_vis, 
                       palette="Set2", ax=axes[0, i], s=50, alpha=0.7)
    else:
        sns.scatterplot(x="PC1", y="PC2", hue=labels, data=df_vis, 
                       palette="tab10", ax=axes[0, i], s=50, alpha=0.7, legend=False)
    axes[0, i].set_title(f"{method} Clustering", fontsize=12, fontweight='bold')
    axes[0, i].set_xlabel("PC1")
    axes[0, i].set_ylabel("PC2")

# Row 2: Hierarchical + Patient batch effect
sns.scatterplot(x="PC1", y="PC2", hue="Hierarchical", data=df_vis, 
               palette="tab10", ax=axes[1, 0], s=50, alpha=0.7, legend=False)
axes[1, 0].set_title("Hierarchical Clustering", fontsize=12, fontweight='bold')

sns.scatterplot(x="PC1", y="PC2", hue="Patient", data=df_vis, 
               palette="viridis", ax=axes[1, 1], s=50, alpha=0.7)
axes[1, 1].set_title("Patient Batch Effect", fontsize=12, fontweight='bold')

# Confusion matrix: True vs Best method
best_method_labels = kmeans_labels if kmeans_ari >= gmm_ari else gmm_labels
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_labels, best_method_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2])
axes[1, 2].set_title("Confusion Matrix\n(True vs Best Method)", fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel("Predicted Cluster")
axes[1, 2].set_ylabel("True Subtype")

plt.tight_layout()
plt.show()

# =============================================================================
# STEP 7: Dendrogram (Hierarchical Only)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: Hierarchical Clustering Dendrogram")
print("=" * 70)

plt.figure(figsize=(12, 5))
dendrogram(linkage_matrix, truncate_mode="level", p=5, color_threshold=0)
plt.axhline(y=linkage_matrix[-optimal_k+1, 2], color='red', linestyle='--', 
            linewidth=2, label=f'Cut at K={optimal_k}')
plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)", 
          fontsize=14, fontweight='bold')
plt.xlabel("Sample Index (or Cluster Size)")
plt.ylabel("Distance")
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# STEP 8: Final Performance Summary
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: Final Performance Summary")
print("=" * 70)

summary_df = pd.DataFrame({
    "Method": ["KMeans", "GaussianMixture", "Hierarchical"],
    "Silhouette": [kmeans_silhouette, gmm_silhouette, hier_silhouette],
    "ARI": [kmeans_ari, gmm_ari, hier_ari]
})

summary_df = summary_df.sort_values("Silhouette", ascending=False).reset_index(drop=True)

print("\nClustering Performance Comparison:")
print(summary_df.to_string(index=False))

best_method = summary_df.iloc[0]["Method"]
best_ari = summary_df.iloc[0]["ARI"]

print(f"\nBest performing method by Silhouette Score: {best_method}")
print(f"Agreement with true subtypes (ARI): {best_ari:.4f}")

if best_ari > 0.7:
    print("  → High agreement: Clustering successfully recovered true subtypes")
elif best_ari > 0.4:
    print("  → Moderate agreement: Clustering partially captured true structure")
else:
    print("  → Low agreement: Patient heterogeneity or batch effects dominate")

print("\n" + "=" * 70)
print("Analysis Complete")
print("=" * 70)
print("\nKey Insights:")
print(f"1. Data simulated realistic scRNA-seq characteristics")
print(f"2. Patient batch effects introduced cross-sample variation")
print(f"3. Optimal K selected as {optimal_k} based on Silhouette Score")
print(f"4. Best method: {best_method} (ARI: {best_ari:.3f})")
