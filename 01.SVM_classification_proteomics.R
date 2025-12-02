# 패키지 로드
library(e1071)  # SVM
library(caret)  # 전처리, 평가
library(pROC)   # ROC, AUC
library(ggplot2)
library(reshape2)
library(gridExtra)

# 재현성을 위한 seed 설정
set.seed(42)

# ============================================
# 프로테오믹스 데이터 생성
# ============================================
n_samples_per_class <- 100
n_proteins <- 1000

# 4개 질병 타입 생성
healthy <- matrix(rnorm(n_samples_per_class * n_proteins), 
                  nrow = n_samples_per_class) + 
           matrix(runif(n_proteins, -0.5, 0.5), 
                  nrow = n_samples_per_class, ncol = n_proteins, byrow = TRUE)

disease_a <- matrix(rnorm(n_samples_per_class * n_proteins), 
                    nrow = n_samples_per_class) + 
             matrix(runif(n_proteins, 0.5, 1.0), 
                    nrow = n_samples_per_class, ncol = n_proteins, byrow = TRUE)

disease_b <- matrix(rnorm(n_samples_per_class * n_proteins), 
                    nrow = n_samples_per_class) + 
             matrix(runif(n_proteins, -1.0, -0.5), 
                    nrow = n_samples_per_class, ncol = n_proteins, byrow = TRUE)

disease_c <- matrix(rnorm(n_samples_per_class * n_proteins), 
                    nrow = n_samples_per_class) + 
             matrix(runif(n_proteins, 1.0, 1.5), 
                    nrow = n_samples_per_class, ncol = n_proteins, byrow = TRUE)

# 데이터 결합
X <- rbind(healthy, disease_a, disease_b, disease_c)
y <- factor(c(rep("Healthy", n_samples_per_class),
              rep("Disease_A", n_samples_per_class),
              rep("Disease_B", n_samples_per_class),
              rep("Disease_C", n_samples_per_class)))

# 데이터프레임 생성
protein_names <- paste0("Protein_", 1:n_proteins)
colnames(X) <- protein_names
df <- data.frame(X, Label = y)

cat("Data Overview:\n")
print(head(df[, c(1:5, ncol(df))]))
cat("\nData shape:", dim(df), "\n")
cat("\nClass distribution:\n")
print(table(df$Label))

# ============================================
# Train-test split
# ============================================
train_index <- createDataPartition(y, p = 0.75, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

cat("\nTrain set:", dim(X_train), "Test set:", dim(X_test), "\n")

# 스케일링
preproc <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preproc, X_train)
X_test_scaled <- predict(preproc, X_test)

# ============================================
# Linear vs RBF Kernel 비교 (GridSearch)
# ============================================
cat("\n=== Comparing Linear vs RBF Kernel ===\n")

# Linear kernel 튜닝
cat("\nTuning Linear Kernel...\n")
tune_linear <- tune.svm(
  x = X_train_scaled,
  y = y_train,
  kernel = "linear",
  cost = c(0.1, 1, 10, 100),
  tunecontrol = tune.control(cross = 5)
)

cat("Best Linear params:\n")
print(tune_linear$best.parameters)
cat("Best Linear CV accuracy:", 1 - tune_linear$best.performance, "\n")

# RBF kernel 튜닝
cat("\nTuning RBF Kernel...\n")
tune_rbf <- tune.svm(
  x = X_train_scaled,
  y = y_train,
  kernel = "radial",
  cost = c(0.1, 1, 10, 100),
  gamma = c(0.001, 0.01, 0.1, 1),
  tunecontrol = tune.control(cross = 5)
)

cat("Best RBF params:\n")
print(tune_rbf$best.parameters)
cat("Best RBF CV accuracy:", 1 - tune_rbf$best.performance, "\n")

# 최적 모델
best_linear <- tune_linear$best.model
best_rbf <- tune_rbf$best.model

# ============================================
# 테스트 세트 성능 비교
# ============================================
cat("\n=== Test Set Performance Comparison ===\n")

# Linear kernel 성능
pred_linear <- predict(best_linear, X_test_scaled)
acc_linear <- mean(pred_linear == y_test)

# RBF kernel 성능
pred_rbf <- predict(best_rbf, X_test_scaled)
acc_rbf <- mean(pred_rbf == y_test)

cat("\nLinear Kernel Test Accuracy:", round(acc_linear, 4), "\n")
cat("RBF Kernel Test Accuracy:", round(acc_rbf, 4), "\n")

# Confusion Matrix
cat("\nConfusion Matrix (Linear):\n")
print(confusionMatrix(pred_linear, y_test))

cat("\nConfusion Matrix (RBF):\n")
print(confusionMatrix(pred_rbf, y_test))

# Multi-class AUC 계산 (One-vs-Rest)
# Linear
pred_prob_linear <- attr(predict(best_linear, X_test_scaled, decision.values = TRUE), 
                         "decision.values")
roc_linear_list <- multiclass.roc(y_test, pred_prob_linear)
auc_linear <- auc(roc_linear_list)

# RBF
pred_prob_rbf <- attr(predict(best_rbf, X_test_scaled, decision.values = TRUE), 
                      "decision.values")
roc_rbf_list <- multiclass.roc(y_test, pred_prob_rbf)
auc_rbf <- auc(roc_rbf_list)

cat("\nLinear Kernel AUC:", round(auc_linear, 4), "\n")
cat("RBF Kernel AUC:", round(auc_rbf, 4), "\n")

# 성능 비교 시각화
performance_df <- data.frame(
  Metric = rep(c("Accuracy", "AUC"), each = 2),
  Kernel = rep(c("Linear", "RBF"), 2),
  Score = c(acc_linear, acc_rbf, auc_linear, auc_rbf)
)

p1 <- ggplot(performance_df, aes(x = Metric, y = Score, fill = Kernel)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = c("Linear" = "skyblue", "RBF" = "lightcoral")) +
  ylim(0, 1) +
  labs(title = "Linear vs RBF Kernel Performance",
       y = "Score") +
  theme_minimal() +
  theme(legend.position = "top")

print(p1)

# ============================================
# Learning Curve 분석
# ============================================
cat("\n=== Learning Curve Analysis ===\n")

# 더 나은 성능의 모델 선택
best_model <- if(acc_rbf >= acc_linear) best_rbf else best_linear
kernel_used <- if(acc_rbf >= acc_linear) "RBF" else "Linear"

cat("Using", kernel_used, "kernel for learning curve analysis...\n")

# Learning curve 계산
train_sizes <- seq(0.1, 1.0, by = 0.1)
train_scores <- numeric(length(train_sizes))
val_scores <- numeric(length(train_sizes))

for(i in seq_along(train_sizes)) {
  size <- round(nrow(X_train_scaled) * train_sizes[i])
  
  # 5-fold CV
  cv_scores_train <- numeric(5)
  cv_scores_val <- numeric(5)
  
  folds <- createFolds(y_train, k = 5, list = TRUE)
  
  for(j in 1:5) {
    val_idx <- folds[[j]]
    train_idx <- setdiff(1:nrow(X_train_scaled), val_idx)
    
    # 서브샘플링
    subsample_idx <- sample(train_idx, min(size, length(train_idx)))
    
    # 모델 학습
    if(kernel_used == "RBF") {
      temp_model <- svm(x = X_train_scaled[subsample_idx, ], 
                       y = y_train[subsample_idx],
                       kernel = "radial",
                       cost = tune_rbf$best.parameters$cost,
                       gamma = tune_rbf$best.parameters$gamma)
    } else {
      temp_model <- svm(x = X_train_scaled[subsample_idx, ], 
                       y = y_train[subsample_idx],
                       kernel = "linear",
                       cost = tune_linear$best.parameters$cost)
    }
    
    # 예측
    pred_train <- predict(temp_model, X_train_scaled[subsample_idx, ])
    pred_val <- predict(temp_model, X_train_scaled[val_idx, ])
    
    cv_scores_train[j] <- mean(pred_train == y_train[subsample_idx])
    cv_scores_val[j] <- mean(pred_val == y_train[val_idx])
  }
  
  train_scores[i] <- mean(cv_scores_train)
  val_scores[i] <- mean(cv_scores_val)
  
  cat("Progress:", round(train_sizes[i] * 100), "% complete\n")
}

# Learning curve 데이터프레임
lc_df <- data.frame(
  TrainSize = rep(round(nrow(X_train_scaled) * train_sizes), 2),
  Score = c(train_scores, val_scores),
  Type = rep(c("Training", "Validation"), each = length(train_sizes))
)

# Learning curve 시각화
p2 <- ggplot(lc_df, aes(x = TrainSize, y = Score, color = Type, group = Type)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Training" = "red", "Validation" = "green")) +
  labs(title = paste("Learning Curve -", kernel_used, "Kernel SVM"),
       x = "Training Set Size",
       y = "Accuracy Score") +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p2)

# Learning curve 해석
final_gap <- train_scores[length(train_scores)] - val_scores[length(val_scores)]
cat("\n=== Learning Curve Interpretation ===\n")
cat("Final training score:", round(train_scores[length(train_scores)], 4), "\n")
cat("Final validation score:", round(val_scores[length(val_scores)], 4), "\n")
cat("Train-Val gap:", round(final_gap, 4), "\n")

if(final_gap > 0.1) {
  cat("→ High variance (overfitting) detected\n")
} else if(val_scores[length(val_scores)] < 0.7) {
  cat("→ High bias (underfitting) detected\n")
} else {
  cat("→ Model shows good balance\n")
}

# ============================================
# ROC Curve (Multi-class)
# ============================================
cat("\n=== ROC Curves ===\n")

# One-vs-Rest ROC for each class
classes <- levels(y_test)
roc_list <- list()

for(class_name in classes) {
  # Binary labels
  binary_test <- ifelse(y_test == class_name, 1, 0)
  
  # Decision values for this class
  decision_col <- which(colnames(pred_prob_rbf) == paste(class_name, "/", sep = ""))
  if(length(decision_col) == 0) {
    decision_col <- grep(class_name, colnames(pred_prob_rbf))[1]
  }
  
  if(length(decision_col) > 0) {
    roc_obj <- roc(binary_test, pred_prob_rbf[, decision_col], quiet = TRUE)
    roc_list[[class_name]] <- roc_obj
    cat(class_name, "AUC:", round(auc(roc_obj), 4), "\n")
  }
}

# ROC curve 시각화
plot(roc_list[[1]], col = "blue", main = "ROC Curves - Multi-class Classification (RBF)")
if(length(roc_list) > 1) lines(roc_list[[2]], col = "red")
if(length(roc_list) > 2) lines(roc_list[[3]], col = "green")
if(length(roc_list) > 3) lines(roc_list[[4]], col = "orange")
abline(a = 0, b = 1, lty = 2, col = "black")
legend("bottomright", 
       legend = names(roc_list),
       col = c("blue", "red", "green", "orange")[1:length(roc_list)],
       lwd = 2)

# ============================================
# 새로운 데이터로 예측
# ============================================
cat("\n=== Predicting on New Samples ===\n")

# 새로운 샘플 생성
new_sample_1 <- matrix(rnorm(n_proteins), nrow = 1) + runif(n_proteins, -0.5, 0.5)
new_sample_2 <- matrix(rnorm(n_proteins), nrow = 1) + runif(n_proteins, 0.5, 1.0)
new_sample_3 <- matrix(rnorm(n_proteins), nrow = 1) + runif(n_proteins, -1.0, -0.5)
new_sample_4 <- matrix(rnorm(n_proteins), nrow = 1) + runif(n_proteins, 1.0, 1.5)
new_sample_5 <- matrix(rnorm(n_proteins), nrow = 1)

new_samples <- rbind(new_sample_1, new_sample_2, new_sample_3, 
                     new_sample_4, new_sample_5)
colnames(new_samples) <- protein_names

# 스케일링
new_samples_scaled <- predict(preproc, new_samples)

# 예측
new_predictions <- predict(best_model, new_samples_scaled)
new_decision <- attr(predict(best_model, new_samples_scaled, decision.values = TRUE), 
                     "decision.values")

cat("\nPredictions using", kernel_used, "kernel:\n")
for(i in 1:nrow(new_samples)) {
  cat("\nSample", i, ":\n")
  cat("  Predicted Class:", as.character(new_predictions[i]), "\n")
  cat("  Decision values:", new_decision[i, ], "\n")
}

# 결과 정리
results_df <- data.frame(
  Sample_ID = paste0("NewSample_", 1:5),
  Predicted_Class = as.character(new_predictions),
  Max_Decision = apply(abs(new_decision), 1, max)
)

cat("\n=== Summary of New Predictions ===\n")
print(results_df)

# 최종 요약
cat("\n=== Final Summary ===\n")
cat("Best performing kernel:", kernel_used, "\n")
cat("Test Accuracy:", round(max(acc_linear, acc_rbf), 4), "\n")
cat("Test AUC:", round(max(auc_linear, auc_rbf), 4), "\n")
