# ==============================================================================
# 1. 패키지 로드 및 설정
# ==============================================================================
library(dplyr)
library(caret)
library(e1071)
library(pROC)
library(ggplot2)
library(tidyr)

set.seed(42) # 난수 시드 설정 (재현성 확보)

# ==============================================================================
# 2. 데이터 생성 및 분할
# ==============================================================================

cat("## 2. 데이터 생성 및 분할\n")

n_samples_per_class <- 100
n_proteins <- 1000
protein_names <- paste0('Protein_', 1:n_proteins)

# 데이터 생성: 4개 클래스 (Healthy, Disease_A, B, C)
healthy <- matrix(rnorm(n_samples_per_class * n_proteins), nrow = n_samples_per_class, ncol = n_proteins) + matrix(runif(n_proteins, -0.5, 0.5), nrow = n_samples_per_class, ncol = n_proteins, byrow = TRUE)
disease_a <- matrix(rnorm(n_samples_per_class * n_proteins), nrow = n_samples_per_class, ncol = n_proteins) + matrix(runif(n_proteins, 0.5, 1.0), nrow = n_samples_per_class, ncol = n_proteins, byrow = TRUE)
disease_b <- matrix(rnorm(n_samples_per_class * n_proteins), nrow = n_samples_per_class, ncol = n_proteins) + matrix(runif(n_proteins, -1.0, -0.5), nrow = n_samples_per_class, ncol = n_proteins, byrow = TRUE)
disease_c <- matrix(rnorm(n_samples_per_class * n_proteins), nrow = n_samples_per_class, ncol = n_proteins) + matrix(runif(n_proteins, 1.0, 1.5), nrow = n_samples_per_class, ncol = n_proteins, byrow = TRUE)

X_data <- rbind(healthy, disease_a, disease_b, disease_c)
y_labels <- factor(c(rep('Healthy', n_samples_per_class), rep('Disease_A', n_samples_per_class), rep('Disease_B', n_samples_per_class), rep('Disease_C', n_samples_per_class)))

df <- as.data.frame(X_data)
colnames(df) <- protein_names
df$Label <- y_labels

# Train-Test Split (클래스 불균형 방지를 위한 stratify 적용)
train_index <- createDataPartition(df$Label, p = 0.75, list = FALSE, times = 1)
train_df <- df[train_index, ]
test_df <- df[-train_index, ]

X_train <- train_df[, -ncol(train_df)]
y_train <- train_df[, ncol(train_df)]
X_test <- test_df[, -ncol(test_df)]
y_test <- test_df[, ncol(test_df)]

cat(sprintf("Train set size: (%d, %d), Test set size: (%d, %d)\n", 
            nrow(X_train), ncol(X_train), nrow(X_test), ncol(X_test)))

# ==============================================================================
# 3. 데이터 전처리: PCA를 이용한 차원 축소
# ==============================================================================

cat("\n## 3. 데이터 전처리: PCA를 이용한 차원 축소\n")

# PCA를 포함하여 scaler 객체 학습 (센터링, 스케일링 후 누적 분산 95% 기준 PCA 적용)
scaler_pca <- preProcess(X_train, method = c("center", "scale", "pca"), thresh = 0.95)

# 주성분 개수 확인
num_components <- scaler_pca$numComp
cat(sprintf("✅ 원본 특성 (1000개) -> 축소된 주성분: %d개\n", num_components))

# 훈련 및 테스트 데이터에 PCA 변환 적용
X_train_pca <- predict(scaler_pca, X_train)
X_test_pca <- predict(scaler_pca, X_test)

cat(sprintf("PCA 변환 후 훈련 데이터셋 크기: (%d, %d)\n", 
            nrow(X_train_pca), ncol(X_train_pca)))

# ==============================================================================
# 4. Kernel SVM 튜닝 (PCA 데이터 사용)
# ==============================================================================

cat("\n## 4. Kernel SVM 튜닝 (PCA 데이터 사용)\n")

# Cross-Validation 설정 (5-fold, 1 repeat, Multi-class 지표 사용)
fit_control <- trainControl(
  method = "repeatedcv", 
  number = 5, 
  repeats = 1, 
  classProbs = TRUE, 
  summaryFunction = multiClassSummary
)

# 4.1. Linear Kernel 튜닝
cat("Tuning Linear Kernel...\n")
grid_linear <- expand.grid(C = c(0.01, 0.1, 1, 10, 100)) 
model_linear_pca <- train(
  x = X_train_pca, 
  y = y_train, 
  method = "svmLinear", 
  trControl = fit_control, 
  tuneGrid = grid_linear,
  preProc = NULL
)
best_C_linear <- model_linear_pca$bestTune$C
best_acc_linear <- max(model_linear_pca$results$Accuracy, na.rm = TRUE)

cat(sprintf("Best Linear PCA (C): %.4f, CV Accuracy: %.4f\n", best_C_linear, best_acc_linear))

# 4.2. RBF Kernel 튜닝
cat("\nTuning RBF Kernel...\n")
grid_rbf <- expand.grid(C = c(0.1, 1, 10, 100), sigma = c(0.001, 0.01, 0.1)) 
model_rbf_pca <- train(
  x = X_train_pca, 
  y = y_train, 
  method = "svmRadial", 
  trControl = fit_control, 
  tuneGrid = grid_rbf,
  preProc = NULL
)
best_C_rbf <- model_rbf_pca$bestTune$C
best_sigma_rbf <- model_rbf_pca$bestTune$sigma
best_acc_rbf <- max(model_rbf_pca$results$Accuracy, na.rm = TRUE)

cat(sprintf("Best RBF PCA (C, sigma): C=%.2f, sigma=%.4f, CV Accuracy: %.4f\n", best_C_rbf, best_sigma_rbf, best_acc_rbf))

# 4.3. 최적 모델 선택
if (best_acc_rbf >= best_acc_linear) {
  best_model <- model_rbf_pca
  kernel_used <- "RBF"
} else {
  best_model <- model_linear_pca
  kernel_used <- "Linear"
}
cat(sprintf("\n>>> 최적 커널: %s Kernel (CV Accuracy: %.4f)\n", kernel_used, max(best_acc_linear, best_acc_rbf)))

# ==============================================================================
# 5. 테스트 세트 성능 평가 및 시각화
# ==============================================================================

cat("\n## 5. 테스트 세트 성능 평가 및 시각화\n")

# 5.1. 테스트 세트 예측 (PCA 데이터 사용)
y_pred_best <- predict(best_model, X_test_pca)
y_prob_best <- predict(best_model, X_test_pca, type = "prob")

# 5.2. 최종 성능 계산
cm_best <- confusionMatrix(y_pred_best, y_test)
acc_test <- cm_best$overall['Accuracy']
auc_test <- multiclass.roc(response = y_test, predictor = y_prob_best, direction = "<")$auc[1]

cat(sprintf("최종 Test Accuracy: %.4f\n", acc_test))
cat(sprintf("최종 Test AUC (macro): %.4f\n", auc_test))

# 5.3. Confusion Matrix 시각화
cm_df <- as.data.frame(cm_best$table)
cm_plot <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "black") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  labs(title = paste("Confusion Matrix -", kernel_used, "Kernel"), x = "True Label", y = "Predicted Label") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(cm_plot)


# 5.4. ROC Curves 시각화 (One-vs-Rest)
roc_data_list <- list()
classes <- levels(y_test)

for (class_name in classes) {
  roc_obj <- roc(response = y_test, predictor = y_prob_best[, class_name], levels = classes, direction = "<")
  roc_data <- data.frame(FPR = 1 - roc_obj$specificities, TPR = roc_obj$sensitivities, Class = class_name, AUC = roc_obj$auc[1])
  roc_data_list[[class_name]] <- roc_data
}
roc_data_df <- bind_rows(roc_data_list)

roc_plot <- ggplot(roc_data_df, aes(x = FPR, y = TPR, group = Class, color = Class)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", linewidth = 1) +
  labs(title = paste('ROC Curves - Multi-class Classification (', kernel_used, ' Kernel)'), x = 'False Positive Rate', y = 'True Positive Rate') +
  theme_minimal() + coord_fixed(ratio = 1) + theme(legend.position = "lower right")
print(roc_plot)
