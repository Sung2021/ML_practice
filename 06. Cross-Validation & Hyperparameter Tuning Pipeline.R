# 06. Cross-Validation & Hyperparameter Tuning Pipeline
# 
# Learning Objectives:
# - Implement various cross-validation strategies
# - Compare hyperparameter tuning methods (Grid, Random)
# - Analyze learning curves and model performance
# - Build reusable tuning pipeline
#
# Datasets:
# - Classification: Cancer subtype prediction (gene expression, 3 subtypes, imbalanced)
# - Regression: Drug response prediction (proteomics, IC50 values, with outliers)

# ============================================================================
# 1. Setup and Load Libraries
# ============================================================================

# Install packages if needed
required_packages <- c("caret", "randomForest", "gbm", "ggplot2", "reshape2", 
                       "gridExtra", "doParallel", "e1071")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

set.seed(42)

# Setup parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# ============================================================================
# 2. Generate Realistic Datasets
# ============================================================================

# Classification: Cancer subtype prediction
n_samples_class <- 500
n_features <- 50

# Generate base features
X_class <- matrix(rnorm(n_samples_class * n_features, mean = 0, sd = 1), 
                  nrow = n_samples_class, ncol = n_features)

# Create 3 cancer subtypes with different signatures
y_class <- sample(c(1, 2, 3), n_samples_class, replace = TRUE, 
                  prob = c(0.5, 0.3, 0.2))  # Imbalanced

# Add subtype-specific signals
for (i in 1:n_samples_class) {
  if (y_class[i] == 1) {
    X_class[i, 1:15] <- X_class[i, 1:15] + rnorm(15, mean = 1, sd = 0.5)
  } else if (y_class[i] == 2) {
    X_class[i, 5:20] <- X_class[i, 5:20] + rnorm(16, mean = 1.5, sd = 0.5)
  } else {
    X_class[i, 10:25] <- X_class[i, 10:25] + rnorm(16, mean = 2, sd = 0.5)
  }
}

# Log transformation (gene expression)
X_class <- abs(X_class) + 0.1
X_class <- log2(X_class + 1)

# Add label noise (5% misdiagnosis)
noise_idx <- sample(1:n_samples_class, size = floor(0.05 * n_samples_class))
y_class[noise_idx] <- sample(c(1, 2, 3), length(noise_idx), replace = TRUE)

# Convert to factor
y_class <- factor(y_class, levels = c(1, 2, 3), 
                  labels = c("Type1", "Type2", "Type3"))

# Create dataframe
colnames(X_class) <- paste0("Gene", 1:n_features)
data_class <- data.frame(X_class, Subtype = y_class)

cat("=== Classification Dataset (Cancer Subtype) ===\n")
cat("Shape:", nrow(data_class), "x", ncol(data_class) - 1, "\n")
cat("Class distribution:\n")
print(table(y_class))
cat("Class proportions:\n")
print(prop.table(table(y_class)))

# Regression: Drug response prediction (IC50)
n_samples_reg <- 400

# Generate base features (proteomics)
X_reg <- matrix(rnorm(n_samples_reg * n_features, mean = 0, sd = 1),
                nrow = n_samples_reg, ncol = n_features)

# Create response based on subset of features
true_coef <- c(rep(2, 5), rep(1.5, 5), rep(1, 2), rep(0, n_features - 12))
y_reg <- X_reg %*% true_coef + rnorm(n_samples_reg, mean = 0, sd = 3)

# Transform to IC50 range (0.1 to 100 μM, log-scale)
y_reg <- (y_reg - min(y_reg)) / (max(y_reg) - min(y_reg))
y_reg <- 0.1 * exp(y_reg * log(1000))

# Add outliers (5%)
n_outliers <- floor(0.05 * n_samples_reg)
outlier_idx <- sample(1:n_samples_reg, n_outliers)
y_reg[outlier_idx] <- y_reg[outlier_idx] * runif(n_outliers, 2, 5)

# Z-score normalize features
X_reg <- scale(X_reg)

# Create dataframe
colnames(X_reg) <- paste0("Protein", 1:n_features)
data_reg <- data.frame(X_reg, IC50 = y_reg)

cat("\n=== Regression Dataset (Drug Response IC50) ===\n")
cat("Shape:", nrow(data_reg), "x", ncol(data_reg) - 1, "\n")
cat("IC50 range:", sprintf("[%.2f, %.2f] μM\n", min(y_reg), max(y_reg)))
cat("IC50 median:", sprintf("%.2f μM\n", median(y_reg)))
cat("IC50 std:", sprintf("%.2f\n", sd(y_reg)))
cat("Number of outliers:", n_outliers, "\n")

# ============================================================================
# 3. Visualize Realistic Data Characteristics
# ============================================================================

# Setup plotting
par(mfrow = c(2, 3))

# Classification: Class distribution
barplot(table(y_class), col = c("#1f77b4", "#ff7f0e", "#2ca02c"),
        main = "Imbalanced Class Distribution",
        xlab = "Cancer Subtype", ylab = "Count",
        names.arg = c("Type 1\n(50%)", "Type 2\n(30%)", "Type 3\n(20%)"))

# Classification: Feature space (first 2 features)
plot(X_class[, 1], X_class[, 2], col = as.numeric(y_class), pch = 19,
     xlab = "Gene 1 (log2 expression)", ylab = "Gene 2 (log2 expression)",
     main = "Feature Space\n(Not Perfectly Separable)")
legend("topright", legend = levels(y_class), col = 1:3, pch = 19, cex = 0.8)

# Classification: Feature correlation heatmap
corr_class <- cor(X_class[, 1:10])
image(1:10, 1:10, corr_class, col = colorRampPalette(c("blue", "white", "red"))(50),
      main = "Feature Correlation\n(Redundant genes)", xlab = "Feature", ylab = "Feature")

# Regression: IC50 distribution
hist(y_reg, breaks = 30, col = "skyblue", border = "black",
     main = "IC50 Distribution\n(with outliers)", xlab = "IC50 (μM)", ylab = "Frequency")
abline(v = median(y_reg), col = "red", lwd = 2, lty = 2)
legend("topright", "Median", col = "red", lwd = 2, lty = 2, cex = 0.8)

# Regression: Log-scale IC50
hist(log10(y_reg), breaks = 30, col = "lightcoral", border = "black",
     main = "Log-transformed IC50\n(Closer to normal)", 
     xlab = "log10(IC50)", ylab = "Frequency")

# Regression: Feature vs Response
plot(X_reg[, 1], y_reg, pch = 19, col = rgb(0, 0, 1, 0.5),
     xlab = "Protein 1 (z-score)", ylab = "IC50 (μM)",
     main = "Feature-Response Relationship", log = "y")
grid()

par(mfrow = c(1, 1))

# ============================================================================
# 4. Cross-Validation Strategies
# ============================================================================

cat("\n=== Cross-Validation Strategies ===\n")

# 4.1 K-Fold Cross-Validation
train_control_kfold <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = TRUE,
  classProbs = TRUE
)

# Train Random Forest with 5-fold CV
rf_kfold <- train(
  Subtype ~ .,
  data = data_class,
  method = "rf",
  trControl = train_control_kfold,
  ntree = 100
)

cat("\nK-Fold Cross-Validation Results:\n")
print(rf_kfold$results)
cat("Mean Accuracy:", sprintf("%.4f\n", rf_kfold$results$Accuracy))

# 4.2 Repeated K-Fold CV
train_control_repeated <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  savePredictions = TRUE,
  classProbs = TRUE
)

rf_repeated <- train(
  Subtype ~ .,
  data = data_class,
  method = "rf",
  trControl = train_control_repeated,
  ntree = 100
)

cat("\nRepeated K-Fold Cross-Validation Results:\n")
print(rf_repeated$results)

# 4.3 Multiple metrics evaluation
train_control_multi <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = TRUE,
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)

rf_multi <- train(
  Subtype ~ .,
  data = data_class,
  method = "rf",
  trControl = train_control_multi,
  metric = "Accuracy",
  ntree = 100
)

cat("\nMultiple Metrics Cross-Validation:\n")
print(rf_multi$results)

# ============================================================================
# 5. Hyperparameter Tuning - Grid Search
# ============================================================================

cat("\n=== Grid Search CV ===\n")

# Split data
train_idx <- createDataPartition(data_class$Subtype, p = 0.8, list = FALSE)
train_data <- data_class[train_idx, ]
test_data <- data_class[-train_idx, ]

# Define parameter grid
tune_grid <- expand.grid(
  mtry = c(5, 10, 15, 20)
)

cat("Total combinations:", nrow(tune_grid), "\n")

# Grid search with caret
train_control_grid <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = TRUE,
  classProbs = TRUE
)

grid_search <- train(
  Subtype ~ .,
  data = train_data,
  method = "rf",
  trControl = train_control_grid,
  tuneGrid = tune_grid,
  ntree = 100
)

cat("\nGrid Search Results:\n")
print(grid_search$results)
cat("\nBest mtry:", grid_search$bestTune$mtry, "\n")
cat("Best CV Accuracy:", sprintf("%.4f\n", max(grid_search$results$Accuracy)))

# Test set performance
test_pred <- predict(grid_search, newdata = test_data)
test_acc <- confusionMatrix(test_pred, test_data$Subtype)$overall["Accuracy"]
cat("Test Accuracy:", sprintf("%.4f\n", test_acc))

# Plot grid search results
plot(grid_search, main = "Grid Search: Accuracy vs mtry")

# ============================================================================
# 6. Hyperparameter Tuning - Random Search
# ============================================================================

cat("\n=== Random Search CV ===\n")

# Random search for GBM
train_control_random <- trainControl(
  method = "cv",
  number = 5,
  search = "random",
  savePredictions = TRUE,
  classProbs = TRUE
)

random_search <- train(
  Subtype ~ .,
  data = train_data,
  method = "rf",
  trControl = train_control_random,
  tuneLength = 10,  # Number of random combinations
  ntree = 100
)

cat("\nRandom Search Results:\n")
print(random_search$results)
cat("\nBest parameters:\n")
print(random_search$bestTune)
cat("Best CV Accuracy:", sprintf("%.4f\n", max(random_search$results$Accuracy)))

# Test set performance
test_pred_random <- predict(random_search, newdata = test_data)
test_acc_random <- confusionMatrix(test_pred_random, test_data$Subtype)$overall["Accuracy"]
cat("Test Accuracy:", sprintf("%.4f\n", test_acc_random))

# Compare Grid vs Random
cat("\n=== Comparison ===\n")
cat("Grid Search   - Best CV:", sprintf("%.4f", max(grid_search$results$Accuracy)),
    ", Test:", sprintf("%.4f\n", test_acc))
cat("Random Search - Best CV:", sprintf("%.4f", max(random_search$results$Accuracy)),
    ", Test:", sprintf("%.4f\n", test_acc_random))

# ============================================================================
# 7. Learning Curves Analysis
# ============================================================================

cat("\n=== Learning Curves Analysis ===\n")

# Function to compute learning curve
compute_learning_curve <- function(data, target_col, train_sizes, model_method = "rf") {
  n_total <- nrow(data)
  train_scores <- numeric(length(train_sizes))
  val_scores <- numeric(length(train_sizes))
  
  for (i in seq_along(train_sizes)) {
    n_train <- floor(train_sizes[i] * n_total)
    
    # Sample subset
    subset_idx <- sample(1:n_total, n_train)
    subset_data <- data[subset_idx, ]
    
    # 5-fold CV on subset
    train_control_lc <- trainControl(method = "cv", number = 5)
    
    model <- train(
      as.formula(paste(target_col, "~ .")),
      data = subset_data,
      method = model_method,
      trControl = train_control_lc,
      ntree = 100,
      tuneGrid = data.frame(mtry = grid_search$bestTune$mtry)
    )
    
    # Train score (refit on full subset)
    train_pred <- predict(model$finalModel, newdata = subset_data)
    train_scores[i] <- mean(train_pred == subset_data[[target_col]])
    
    # Validation score (from CV)
    val_scores[i] <- model$results$Accuracy
    
    cat(sprintf("Train size: %d, Train acc: %.4f, Val acc: %.4f\n",
                n_train, train_scores[i], val_scores[i]))
  }
  
  return(list(train_sizes = train_sizes, 
              train_scores = train_scores, 
              val_scores = val_scores))
}

# Compute learning curve
train_sizes <- seq(0.2, 1.0, by = 0.2)
lc_results <- compute_learning_curve(train_data, "Subtype", train_sizes)

# Plot learning curve
plot(lc_results$train_sizes * nrow(train_data), lc_results$train_scores, 
     type = "b", col = "blue", pch = 19, ylim = c(0.5, 1.0),
     xlab = "Training Set Size", ylab = "Accuracy",
     main = "Learning Curve - Training vs Validation Score")
lines(lc_results$train_sizes * nrow(train_data), lc_results$val_scores, 
      type = "b", col = "red", pch = 17)
legend("bottomright", legend = c("Training", "Validation"), 
       col = c("blue", "red"), pch = c(19, 17), lty = 1)
grid()

# Diagnosis
gap <- tail(lc_results$train_scores, 1) - tail(lc_results$val_scores, 1)
cat("\nLearning Curve Diagnosis:\n")
cat("Final training accuracy:", sprintf("%.4f\n", tail(lc_results$train_scores, 1)))
cat("Final validation accuracy:", sprintf("%.4f\n", tail(lc_results$val_scores, 1)))
cat("Gap:", sprintf("%.4f\n", gap))

if (gap > 0.1) {
  cat("→ High variance (overfitting) - consider regularization or more data\n")
} else if (tail(lc_results$val_scores, 1) < 0.7) {
  cat("→ High bias (underfitting) - consider more complex model or features\n")
} else {
  cat("→ Good fit - model generalizes well\n")
}

# ============================================================================
# 8. Pipeline Integration (Preprocessing + Model)
# ============================================================================

cat("\n=== Pipeline Integration ===\n")

# Create preprocessing object
preProcess_params <- c("center", "scale")

# Train with preprocessing
train_control_pipe <- trainControl(
  method = "cv",
  number = 5,
  preProcOptions = list(center = TRUE, scale = TRUE)
)

pipeline_model <- train(
  Subtype ~ .,
  data = train_data,
  method = "rf",
  trControl = train_control_pipe,
  preProcess = preProcess_params,
  tuneGrid = data.frame(mtry = grid_search$bestTune$mtry),
  ntree = 100
)

cat("\nPipeline Model Results:\n")
print(pipeline_model$results)

# Test performance
test_pred_pipe <- predict(pipeline_model, newdata = test_data)
test_acc_pipe <- confusionMatrix(test_pred_pipe, test_data$Subtype)$overall["Accuracy"]
cat("Test Accuracy with Pipeline:", sprintf("%.4f\n", test_acc_pipe))

# ============================================================================
# 9. Regression Problem Tuning
# ============================================================================

cat("\n=== Regression Problem Tuning ===\n")

# Split regression data
train_idx_reg <- createDataPartition(data_reg$IC50, p = 0.8, list = FALSE)
train_data_reg <- data_reg[train_idx_reg, ]
test_data_reg <- data_reg[-train_idx_reg, ]

# Define GBM grid
gbm_grid <- expand.grid(
  n.trees = c(50, 100, 200),
  interaction.depth = c(3, 5, 7),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = 10
)

cat("Total combinations:", nrow(gbm_grid), "\n")

# Grid search for regression
train_control_reg <- trainControl(
  method = "cv",
  number = 5
)

grid_search_reg <- train(
  IC50 ~ .,
  data = train_data_reg,
  method = "gbm",
  trControl = train_control_reg,
  tuneGrid = gbm_grid,
  verbose = FALSE
)

cat("\nRegression Grid Search Results:\n")
print(grid_search_reg$results[1:5, ])
cat("\nBest parameters:\n")
print(grid_search_reg$bestTune)

# Test set performance
test_pred_reg <- predict(grid_search_reg, newdata = test_data_reg)
test_mse <- mean((test_pred_reg - test_data_reg$IC50)^2)
test_rmse <- sqrt(test_mse)

cat("\nTest MSE:", sprintf("%.2f\n", test_mse))
cat("Test RMSE:", sprintf("%.2f\n", test_rmse))

# Visualize predictions
par(mfrow = c(1, 2))

# Predictions vs True
plot(test_data_reg$IC50, test_pred_reg, pch = 19, col = rgb(0, 0, 1, 0.5),
     xlab = "True IC50", ylab = "Predicted IC50",
     main = sprintf("Predictions (RMSE: %.2f)", test_rmse))
abline(0, 1, col = "red", lwd = 2, lty = 2)
grid()

# Residuals
residuals <- test_data_reg$IC50 - test_pred_reg
plot(test_pred_reg, residuals, pch = 19, col = rgb(0, 0, 1, 0.5),
     xlab = "Predicted IC50", ylab = "Residuals",
     main = "Residual Plot")
abline(h = 0, col = "red", lwd = 2, lty = 2)
grid()

par(mfrow = c(1, 1))

# ============================================================================
# 10. Nested Cross-Validation
# ============================================================================

cat("\n=== Nested Cross-Validation ===\n")

# Outer CV
outer_folds <- createFolds(train_data$Subtype, k = 5, list = TRUE)
nested_scores <- numeric(5)

for (i in seq_along(outer_folds)) {
  cat(sprintf("\nOuter fold %d/%d\n", i, length(outer_folds)))
  
  # Split outer fold
  outer_test_idx <- outer_folds[[i]]
  outer_train <- train_data[-outer_test_idx, ]
  outer_test <- train_data[outer_test_idx, ]
  
  # Inner CV for hyperparameter tuning
  train_control_inner <- trainControl(
    method = "cv",
    number = 3,
    savePredictions = FALSE
  )
  
  inner_model <- train(
    Subtype ~ .,
    data = outer_train,
    method = "rf",
    trControl = train_control_inner,
    tuneGrid = tune_grid,
    ntree = 100
  )
  
  # Evaluate on outer test fold
  outer_pred <- predict(inner_model, newdata = outer_test)
  nested_scores[i] <- mean(outer_pred == outer_test$Subtype)
  cat(sprintf("Outer fold %d accuracy: %.4f\n", i, nested_scores[i]))
}

cat("\n=== Nested CV Results ===\n")
cat("Nested CV scores:", sprintf("%.4f ", nested_scores), "\n")
cat("Mean score:", sprintf("%.4f (+/- %.4f)\n", mean(nested_scores), sd(nested_scores)))

# Compare with non-nested CV
cat("\n=== Non-Nested vs Nested CV Comparison ===\n")
cat("Non-nested CV score:", sprintf("%.4f (optimistically biased)\n", 
                                    max(grid_search$results$Accuracy)))
cat("Nested CV score:    ", sprintf("%.4f (unbiased estimate)\n", mean(nested_scores)))
cat("Bias:               ", sprintf("%.4f\n", 
                                    max(grid_search$results$Accuracy) - mean(nested_scores)))
cat("\nNested CV provides more realistic performance expectation on unseen data\n")

# ============================================================================
# 11. Summary and Best Practices
# ============================================================================

cat("\n=== Summary and Best Practices ===\n\n")

cat("Cross-Validation:\n")
cat("- Use stratified sampling for classification (createDataPartition)\n")
cat("- 5-10 folds typical, balance bias/variance and computation\n")
cat("- Use repeatedcv for more stable estimates\n\n")

cat("Hyperparameter Tuning:\n")
cat("- Grid Search: exhaustive but expensive (expand.grid)\n")
cat("- Random Search: efficient for large spaces (search='random')\n")
cat("- Consider computational cost vs performance gain\n\n")

cat("Pipeline Integration:\n")
cat("- Use preProcess in caret to prevent data leakage\n")
cat("- Scaling/encoding happens within CV folds\n")
cat("- Ensures fair performance estimation\n\n")

cat("Learning Curves:\n")
cat("- Diagnose overfitting (high gap) vs underfitting (low performance)\n")
cat("- Determine if more data would help\n\n")

cat("Nested Cross-Validation:\n")
cat("- Outer loop: unbiased performance estimation\n")
cat("- Inner loop: hyperparameter tuning\n")
cat("- Provides realistic expectation on unseen data\n\n")

# Cleanup
stopCluster(cl)
registerDoSEQ()

cat("\nAnalysis complete!\n")
