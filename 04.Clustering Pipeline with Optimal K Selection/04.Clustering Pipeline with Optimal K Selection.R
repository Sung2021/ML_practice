library(ggplot2)
library(dplyr)
library(tidyr)
library(cluster)
library(mclust)
library(factoextra)
library(gridExtra)

set.seed(42)

# =============================================================================
# STEP 1: Generate Realistic scRNA-seq Style Cancer Data
# =============================================================================
cat("======================================================================\n")
cat("STEP 1: Realistic scRNA-seq Cancer Data Generation\n")
cat("======================================================================\n")

n_samples_per_subtype <- c(180, 120, 100)
n_samples <- sum(n_samples_per_subtype)
n_genes <- 2000
n_patients <- 6

cat(sprintf("Simulating cancer subtypes:\n"))
cat(sprintf("  - Subtype 1 (Proliferative): %d cells\n", n_samples_per_subtype[1]))
cat(sprintf("  - Subtype 2 (Invasive): %d cells\n", n_samples_per_subtype[2]))
cat(sprintf("  - Subtype 3 (Inflammatory): %d cells\n", n_samples_per_subtype[3]))
cat(sprintf("Total cells: %d\n", n_samples))
cat(sprintf("Number of genes: %d\n", n_genes))
cat(sprintf("Number of patients: %d\n", n_patients))

# Subtype 1: Proliferative state
X1_base <- matrix(rnbinom(n_samples_per_subtype[1] * n_genes, size = 5, prob = 0.3), 
                  nrow = n_samples_per_subtype[1], ncol = n_genes)
X1_signature <- matrix(0, nrow = n_samples_per_subtype[1], ncol = n_genes)
X1_signature[, 1:200] <- matrix(rnbinom(n_samples_per_subtype[1] * 200, size = 10, prob = 0.2),
                                nrow = n_samples_per_subtype[1], ncol = 200)
X1 <- X1_base + X1_signature

# Subtype 2: Invasive state
X2_base <- matrix(rnbinom(n_samples_per_subtype[2] * n_genes, size = 5, prob = 0.3),
                  nrow = n_samples_per_subtype[2], ncol = n_genes)
X2_signature <- matrix(0, nrow = n_samples_per_subtype[2], ncol = n_genes)
X2_signature[, 201:400] <- matrix(rnbinom(n_samples_per_subtype[2] * 200, size = 12, prob = 0.15),
                                  nrow = n_samples_per_subtype[2], ncol = 200)
X2 <- X2_base + X2_signature

# Subtype 3: Inflammatory state
X3_base <- matrix(rnbinom(n_samples_per_subtype[3] * n_genes, size = 5, prob = 0.3),
                  nrow = n_samples_per_subtype[3], ncol = n_genes)
X3_signature <- matrix(0, nrow = n_samples_per_subtype[3], ncol = n_genes)
X3_signature[, 401:600] <- matrix(rnbinom(n_samples_per_subtype[3] * 200, size = 15, prob = 0.1),
                                  nrow = n_samples_per_subtype[3], ncol = 200)
X3 <- X3_base + X3_signature

X <- rbind(X1, X2, X3)

# Add patient-specific batch effects
patient_ids <- rep(1:n_patients, each = n_samples / n_patients)
for (patient in 1:n_patients) {
  patient_mask <- patient_ids == patient
  batch_effect <- matrix(rnorm(n_genes, mean = 0, sd = 0.3), nrow = 1, ncol = n_genes)
  X[patient_mask, ] <- X[patient_mask, ] + matrix(rep(batch_effect, sum(patient_mask)), 
                                                   nrow = sum(patient_mask), byrow = TRUE)
}

# Add dropout events
dropout_rate <- 0.15
dropout_mask <- matrix(runif(n_samples * n_genes) < dropout_rate, nrow = n_samples, ncol = n_genes)
X[dropout_mask] <- 0

# Add missing data
missing_rate <- 0.05
missing_mask <- matrix(runif(n_samples * n_genes) < missing_rate, nrow = n_samples, ncol = n_genes)
X[missing_mask] <- NA

# Impute missing values with gene mean
col_means <- colMeans(X, na.rm = TRUE)
for (j in 1:ncol(X)) {
  X[is.na(X[, j]), j] <- col_means[j]
}

# Log transformation
X <- log1p(X)

# Create labels
true_labels <- c(rep(0, n_samples_per_subtype[1]),
                rep(1, n_samples_per_subtype[2]),
                rep(2, n_samples_per_subtype[3]))

# Create dataframe
gene_names <- paste0("Gene_", 1:n_genes)
colnames(X) <- gene_names

df <- as.data.frame(X)
df$True_Subtype <- true_labels
df$Patient_ID <- patient_ids
df$Subtype_Name <- factor(true_labels, levels = 0:2, 
                          labels = c("Proliferative", "Invasive", "Inflammatory"))

cat(sprintf("\nData characteristics:\n"))
cat(sprintf("  - Zero expression rate: %.1f%%\n", sum(X == 0) / length(X) * 100))
cat(sprintf("  - Mean expression: %.3f\n", mean(X)))
cat(sprintf("  - Std expression: %.3f\n", sd(X)))

cat("\nSubtype distribution:\n")
print(table(df$Subtype_Name))

cat("\nPatient distribution:\n")
print(table(df$Patient_ID, df$Subtype_Name))

# =============================================================================
# STEP 2: Data Scaling
# =============================================================================
cat("\n======================================================================\n")
cat("STEP 2: Data Scaling\n")
cat("======================================================================\n")

X_scaled <- scale(X)

cat(sprintf("Scaling completed\n"))
cat(sprintf("Mean of scaled data: %.6f\n", mean(X_scaled)))
cat(sprintf("Std of scaled data: %.6f\n", sd(X_scaled)))

# =============================================================================
# STEP 3: Determine Optimal Number of Clusters
# =============================================================================
cat("\n======================================================================\n")
cat("STEP 3: Optimal Cluster Number Selection\n")
cat("======================================================================\n")

k_range <- 2:10
inertias <- numeric(length(k_range))
silhouettes <- numeric(length(k_range))
bics <- numeric(length(k_range))

cat("Testing cluster numbers from 2 to 10...\n")
for (i in seq_along(k_range)) {
  k <- k_range[i]
  
  # KMeans for Elbow Method
  kmeans_temp <- kmeans(X_scaled, centers = k, nstart = 10)
  inertias[i] <- kmeans_temp$tot.withinss
  silhouettes[i] <- mean(silhouette(kmeans_temp$cluster, dist(X_scaled))[, 3])
  
  # GMM for BIC
  gmm_temp <- Mclust(X_scaled, G = k, verbose = FALSE)
  bics[i] <- gmm_temp$bic
}

optimal_k_df <- data.frame(
  K = k_range,
  Inertia = inertias,
  Silhouette = silhouettes,
  BIC = bics
)

cat("\nOptimal K Selection Results:\n")
print(optimal_k_df, row.names = FALSE)

optimal_k_silhouette <- optimal_k_df$K[which.max(optimal_k_df$Silhouette)]
optimal_k_bic <- optimal_k_df$K[which.min(optimal_k_df$BIC)]

cat(sprintf("\nRecommended K by Silhouette Score: %d\n", optimal_k_silhouette))
cat(sprintf("Recommended K by BIC: %d\n", optimal_k_bic))

# Visualization of optimal K
p1 <- ggplot(optimal_k_df, aes(x = K, y = Inertia)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  labs(title = "Elbow Method", 
       x = "Number of Clusters (K)", 
       y = "Inertia (Within-cluster Sum of Squares)") +
  theme_minimal()

p2 <- ggplot(optimal_k_df, aes(x = K, y = Silhouette)) +
  geom_line(linewidth = 1, color = "orange") +
  geom_point(size = 3, color = "orange") +
  geom_vline(xintercept = optimal_k_silhouette, linetype = "dashed", 
             color = "red", linewidth = 1) +
  annotate("text", x = optimal_k_silhouette, y = max(silhouettes) * 0.95,
           label = paste("Optimal K =", optimal_k_silhouette), hjust = -0.1) +
  labs(title = "Silhouette Analysis",
       x = "Number of Clusters (K)",
       y = "Silhouette Score") +
  theme_minimal()

p3 <- ggplot(optimal_k_df, aes(x = K, y = BIC)) +
  geom_line(linewidth = 1, color = "green") +
  geom_point(size = 3, color = "green") +
  geom_vline(xintercept = optimal_k_bic, linetype = "dashed",
             color = "red", linewidth = 1) +
  annotate("text", x = optimal_k_bic, y = max(bics) * 0.95,
           label = paste("Optimal K =", optimal_k_bic), hjust = -0.1) +
  labs(title = "Bayesian Information Criterion",
       x = "Number of Clusters (K)",
       y = "BIC (Lower is Better)") +
  theme_minimal()

grid.arrange(p1, p2, p3, ncol = 3)

# =============================================================================
# STEP 4: Silhouette Plot for Optimal K
# =============================================================================
cat("\n======================================================================\n")
cat("STEP 4: Detailed Silhouette Analysis for Optimal K\n")
cat("======================================================================\n")

optimal_k <- optimal_k_silhouette
kmeans_optimal <- kmeans(X_scaled, centers = optimal_k, nstart = 10)
optimal_labels <- kmeans_optimal$cluster

sil <- silhouette(optimal_labels, dist(X_scaled))
silhouette_avg <- mean(sil[, 3])

cat(sprintf("Average Silhouette Score for K=%d: %.4f\n", optimal_k, silhouette_avg))

fviz_silhouette(sil) +
  labs(title = paste("Silhouette Plot for K =", optimal_k)) +
  theme_minimal()

# =============================================================================
# STEP 5: Apply Three Clustering Methods with Optimal K
# =============================================================================
cat("\n======================================================================\n")
cat("STEP 5: Apply Clustering Methods\n")
cat("======================================================================\n")

# KMeans
kmeans_result <- kmeans(X_scaled, centers = optimal_k, nstart = 10)
kmeans_labels <- kmeans_result$cluster
kmeans_silhouette <- mean(silhouette(kmeans_labels, dist(X_scaled))[, 3])
kmeans_ari <- adjustedRandIndex(true_labels, kmeans_labels)

cat(sprintf("\nKMeans (K=%d):\n", optimal_k))
cat(sprintf("  Silhouette Score: %.4f\n", kmeans_silhouette))
cat(sprintf("  ARI vs True Labels: %.4f\n", kmeans_ari))

# Gaussian Mixture Model
gmm_result <- Mclust(X_scaled, G = optimal_k, verbose = FALSE)
gmm_labels <- gmm_result$classification
gmm_silhouette <- mean(silhouette(gmm_labels, dist(X_scaled))[, 3])
gmm_ari <- adjustedRandIndex(true_labels, gmm_labels)

cat(sprintf("\nGaussian Mixture Model (K=%d):\n", optimal_k))
cat(sprintf("  Silhouette Score: %.4f\n", gmm_silhouette))
cat(sprintf("  ARI vs True Labels: %.4f\n", gmm_ari))

# Hierarchical Clustering
hc <- hclust(dist(X_scaled), method = "ward.D2")
hierarchical_labels <- cutree(hc, k = optimal_k)
hier_silhouette <- mean(silhouette(hierarchical_labels, dist(X_scaled))[, 3])
hier_ari <- adjustedRandIndex(true_labels, hierarchical_labels)

cat(sprintf("\nHierarchical Clustering (K=%d):\n", optimal_k))
cat(sprintf("  Silhouette Score: %.4f\n", hier_silhouette))
cat(sprintf("  ARI vs True Labels: %.4f\n", hier_ari))

# =============================================================================
# STEP 6: PCA Visualization
# =============================================================================
cat("\n======================================================================\n")
cat("STEP 6: PCA Visualization\n")
cat("======================================================================\n")

pca_result <- prcomp(X_scaled, center = FALSE, scale. = FALSE)
variance_explained <- summary(pca_result)$importance[2, 1:2]

cat(sprintf("Explained variance by PC1: %.2f%%\n", variance_explained[1] * 100))
cat(sprintf("Explained variance by PC2: %.2f%%\n", variance_explained[2] * 100))
cat(sprintf("Total explained variance: %.2f%%\n", sum(variance_explained) * 100))

df_vis <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  True = df$Subtype_Name,
  KMeans = factor(kmeans_labels),
  GMM = factor(gmm_labels),
  Hierarchical = factor(hierarchical_labels),
  Patient = factor(patient_ids)
)

p1 <- ggplot(df_vis, aes(x = PC1, y = PC2, color = True)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(title = "True Subtypes Clustering", x = "PC1", y = "PC2") +
  theme_minimal() +
  theme(legend.position = "right")

p2 <- ggplot(df_vis, aes(x = PC1, y = PC2, color = KMeans)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(title = "KMeans Clustering", x = "PC1", y = "PC2") +
  theme_minimal() +
  theme(legend.position = "none")

p3 <- ggplot(df_vis, aes(x = PC1, y = PC2, color = GMM)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(title = "GMM Clustering", x = "PC1", y = "PC2") +
  theme_minimal() +
  theme(legend.position = "none")

p4 <- ggplot(df_vis, aes(x = PC1, y = PC2, color = Hierarchical)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(title = "Hierarchical Clustering", x = "PC1", y = "PC2") +
  theme_minimal() +
  theme(legend.position = "none")

p5 <- ggplot(df_vis, aes(x = PC1, y = PC2, color = Patient)) +
  geom_point(size = 2, alpha = 0.7) +
  scale_color_viridis_d() +
  labs(title = "Patient Batch Effect", x = "PC1", y = "PC2") +
  theme_minimal() +
  theme(legend.position = "right")

# Confusion matrix
best_method_labels <- if (kmeans_ari >= gmm_ari) kmeans_labels else gmm_labels
cm <- table(True = true_labels, Predicted = best_method_labels)
cm_df <- as.data.frame(cm)

p6 <- ggplot(cm_df, aes(x = Predicted, y = True, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix\n(True vs Best Method)",
       x = "Predicted Cluster", y = "True Subtype") +
  theme_minimal() +
  theme(legend.position = "none")

grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 3)

# =============================================================================
# STEP 7: Dendrogram (Hierarchical Only)
# =============================================================================
cat("\n======================================================================\n")
cat("STEP 7: Hierarchical Clustering Dendrogram\n")
cat("======================================================================\n")

plot(hc, main = "Hierarchical Clustering Dendrogram (Ward Linkage)",
     xlab = "Sample Index", ylab = "Distance", cex = 0.3)
abline(h = hc$height[length(hc$height) - optimal_k + 2], col = "red", 
       lwd = 2, lty = 2)
legend("topright", legend = paste("Cut at K =", optimal_k), 
       col = "red", lty = 2, lwd = 2)

# =============================================================================
# STEP 8: Final Performance Summary
# =============================================================================
cat("\n======================================================================\n")
cat("STEP 8: Final Performance Summary\n")
cat("======================================================================\n")

summary_df <- data.frame(
  Method = c("KMeans", "GaussianMixture", "Hierarchical"),
  Silhouette = c(kmeans_silhouette, gmm_silhouette, hier_silhouette),
  ARI = c(kmeans_ari, gmm_ari, hier_ari)
)

summary_df <- summary_df[order(-summary_df$Silhouette), ]

cat("\nClustering Performance Comparison:\n")
print(summary_df, row.names = FALSE)

best_method <- summary_df$Method[1]
best_ari <- summary_df$ARI[1]

cat(sprintf("\nBest performing method by Silhouette Score: %s\n", best_method))
cat(sprintf("Agreement with true subtypes (ARI): %.4f\n", best_ari))

if (best_ari > 0.7) {
  cat("  → High agreement: Clustering successfully recovered true subtypes\n")
} else if (best_ari > 0.4) {
  cat("  → Moderate agreement: Clustering partially captured true structure\n")
} else {
  cat("  → Low agreement: Patient heterogeneity or batch effects dominate\n")
}

cat("\n======================================================================\n")
cat("Analysis Complete\n")
cat("======================================================================\n")
cat("\nKey Insights:\n")
cat("1. Data simulated realistic scRNA-seq characteristics\n")
cat("2. Patient batch effects introduced cross-sample variation\n")
cat(sprintf("3. Optimal K selected as %d based on Silhouette Score\n", optimal_k))
cat(sprintf("4. Best method: %s (ARI: %.3f)\n", best_method, best_ari))
