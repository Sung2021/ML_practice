# ============================================
# 1. Load Libraries
# ============================================
# 필요한 라이브러리 설치 및 로드
# install.packages(c("tidyverse", "caret", "xgboost", "shapviz", "Matrix"))

library(tidyverse) # 데이터 처리 및 시각화 (dplyr, ggplot2 포함)
library(caret)     # 데이터 분할 및 하이퍼파라미터 튜닝
library(xgboost)   # XGBoost 모델링
library(shapviz)   # SHAP 분석 및 시각화
library(Matrix)    # 희소 행렬 처리 (xgboost 입력 형식)
library(ggplot2)   # 시각화 기본 설정

# 시각화 설정
theme_set(theme_minimal())
set.seed(42)

# ============================================
# 2. 임상 데이터 생성 및 비선형성 부여
# ============================================
cat("Data Overview (데이터 개요):\n")

n_samples <- 500
n_features <- 30
feature_names <- paste0('Clinical_Feature_', 1:n_features)

# 임상 특징 생성 (평균 0, 표준편차 1)
X <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)
colnames(X) <- feature_names

# 실제 환경 시뮬레이션: 타겟 변수에 복잡한 관계 부여
true_weights <- rnorm(n_features) * 2

# 1. 상호작용 항 (Interaction Term): Feature 1과 Feature 6의 곱
interaction_term <- X[, 1] * X[, 6] * 5

# 2. 비선형 항 (Nonlinear Term): Feature 11의 지수 함수
nonlinear_term <- exp(X[, 11] / 3) * 4

# 3. 중요하지 않은 특징의 가중치
noise_features_term <- rowSums(X[, 21:30]) * 0.5 

# 치료 반응 점수 생성 (0-100 사이, 복잡한 특징 관계 포함)
y <- (X %*% true_weights * 2 + interaction_term + nonlinear_term + noise_features_term +
        rnorm(n_samples, sd = 8)) # 노이즈 레벨 증가

# 0-100 범위로 스케일링 (Response Score)
y <- (y - min(y)) / (max(y) - min(y)) * 100

# 데이터프레임 생성
df <- as.data.frame(X)
df$Response_Score <- y

print(head(df))
cat(sprintf("\nData shape (데이터 형태): %d rows, %d columns\n", nrow(df), ncol(df)))
cat("\nResponse Score statistics (반응 점수 통계):\n")
print(summary(df$Response_Score))

# ============================================
# 3. Train-test split & Data Preprocessing (데이터 전처리)
# ============================================
# 데이터 분할
train_index <- createDataPartition(df$Response_Score, p = 0.8, list = FALSE)
df_train <- df[train_index, ]
df_test <- df[-train_index, ]

# 특성(X)과 타겟(Y) 분리
X_train <- as.matrix(df_train[, -ncol(df_train)])
y_train <- df_train$Response_Score
X_test <- as.matrix(df_test[, -ncol(df_test)])
y_test <- df_test$Response_Score

# StandardScaler 적용을 위한 preProcess 객체 생성
preproc <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preproc, X_train)
X_test_scaled <- predict(preproc, X_test)

# xgboost DMatrix 형식으로 변환 (효율적인 학습을 위해)
dtrain <- xgb.DMatrix(data = X_train_scaled, label = y_train)
dtest <- xgb.DMatrix(data = X_test_scaled, label = y_test)

cat(sprintf("\nTrain set (훈련 세트): %d rows, Test set (테스트 세트): %d rows\n", nrow(X_train_scaled), nrow(X_test_scaled)))
cat("Data has been scaled (데이터 스케일링 완료).\n")

# ============================================
# 4. Hyperparameter Tuning with Caret (하이퍼파라미터 튜닝)
# ============================================
cat("\n=== Hyperparameter Tuning (하이퍼파라미터 튜닝 시작) ===\n")
start_time <- Sys.time()

# 튜닝 그리드 설정 (Python 코드와 유사하게 L1/L2 정규화 포함)
tune_grid <- expand.grid(
  nrounds = c(150, 300), 
  max_depth = c(3, 5), 
  eta = c(0.05, 0.1), # learning_rate
  gamma = 0,         # 일반적으로 튜닝하지 않거나 낮은 값으로 시작
  colsample_bytree = 0.8, 
  min_child_weight = 1, # Python 코드에는 없었지만, XGBoost의 핵심 파라미터이므로 추가
  subsample = 0.8,
  lambda = c(0.1, 1, 5),    # reg_lambda
  alpha = c(0, 0.1, 0.5)    # reg_alpha
)

# Caret의 train 함수를 사용하여 Grid Search 수행
# 'xgbTree' 모델은 내부적으로 nrounds(n_estimators) 파라미터를 사용함
model_tuned_caret <- train(
  x = X_train_scaled, 
  y = y_train, 
  method = "xgbTree", 
  tuneGrid = tune_grid,
  trControl = trainControl(method = "cv", number = 5, verboseIter = FALSE), # 5-Fold Cross-Validation
  metric = "RMSE",
  verbose = FALSE 
)

end_time <- Sys.time()

cat(sprintf("\nGrid Search Completed in %.2f seconds.\n", as.numeric(end_time - start_time, units = "secs")))
cat("Best parameters (최적 파라미터):\n")
print(model_tuned_caret$bestTune)
cat(sprintf("Best CV score (최적 CV 점수, RMSE): %.4f\n", min(model_tuned_caret$results$RMSE)))

# 최적 모델 추출
model_tuned <- model_tuned_caret$finalModel

# 예측
y_pred_test_tuned <- predict(model_tuned, dtest)

# 성능 평가
test_rmse_tuned <- RMSE(y_pred_test_tuned, y_test)
test_mae_tuned <- MAE(y_pred_test_tuned, y_test)
test_r2_tuned <- R2(y_pred_test_tuned, y_test)

cat("\n=== Tuned Model Test Performance (최적 모델 테스트 성능) ===\n")
cat(sprintf("Test RMSE: %.4f\n", test_rmse_tuned))
cat(sprintf("Test MAE: %.4f\n", test_mae_tuned))
cat(sprintf("Test R²: %.4f\n", test_r2_tuned))


# ============================================
# 5. Feature Importance 분석 및 시각화
# ============================================
cat("\n=== Feature Importance Analysis (특성 중요도 분석) ===\n")

# XGBoost 내장 feature importance
importance_matrix <- xgb.importance(model = model_tuned)

cat("\nTop 10 Most Important Features (가장 중요한 특성 10개):\n")
print(head(importance_matrix, 10))

# Feature importance 시각화
xgb.plot.importance(importance_matrix[1:15,], top_n = 15, col = "darkred", 
                    main = "Top 15 Feature Importances (상위 15개 특성 중요도)")

# ============================================
# 6. SHAP 분석 (모델 해석력 강화)
# ============================================
cat("\n=== SHAP Analysis (SHAP 분석) ===\n")

# SHAP values 계산 (테스트 세트)
shap_result <- shap.values(model_tuned, X_test_scaled)
shap_vals <- shap_result$shap_values

# 1. SHAP summary plot (전역적 특성 중요도 및 영향 방향)
cat("\nSHAP Summary Plot: Global Feature Influence\n")
sv <- shapviz(shap_vals, X_test_scaled)
sv_plot <- sv_importance(sv, kind = "beeswarm") + 
  ggtitle("SHAP Summary Plot") +
  theme(plot.title = element_text(hjust = 0.5))

print(sv_plot)

# 2. SHAP dependence plot (상위 1개 특성의 영향 관계 시각화)
top_feature_name <- importance_matrix$Feature[1]

cat(sprintf("\nSHAP Dependence Plot for Top Feature: %s\n", top_feature_name))

sv_dep_plot <- sv_dependence(sv, v = top_feature_name) +
  ggtitle(paste0('SHAP Dependence Plot for ', top_feature_name)) +
  theme(plot.title = element_text(hjust = 0.5))

print(sv_dep_plot)

# ============================================
# 7. 예측 vs 실제 시각화
# ============================================
plot_df <- data.frame(
  True_Score = y_test,
  Predicted_Score = y_pred_test_tuned,
  Residuals = y_test - y_pred_test_tuned
)

p1 <- ggplot(plot_df, aes(x = True_Score, y = Predicted_Score)) +
  geom_point(alpha = 0.6, color = 'blue') +
  geom_abline(intercept = 0, slope = 1, color = 'red', linetype = 'dashed', linewidth = 1) +
  labs(
    x = 'True Response Score (실제 반응 점수)', 
    y = 'Predicted Response Score (예측 반응 점수)',
    title = paste0('Predictions vs True Values (예측 대 실제, R²=', round(test_r2_tuned, 3), ')')
  ) +
  theme_minimal() +
  coord_fixed()

p2 <- ggplot(plot_df, aes(x = Predicted_Score, y = Residuals)) +
  geom_point(alpha = 0.6, color = 'darkgreen') +
  geom_hline(yintercept = 0, color = 'red', linewidth = 1) +
  labs(
    x = 'Predicted Response Score (예측 반응 점수)', 
    y = 'Residuals (잔차)',
    title = 'Residual Plot (잔차 플롯)'
  ) +
  theme_minimal()

# 두 플롯을 나란히 표시
cowplot::plot_grid(p1, p2, ncol = 2)


# ============================================
# 8. 새로운 환자 데이터로 예측 및 개별 해석
# ============================================
cat("\n=== Predicting on New Patients (새로운 환자 예측 및 개별 해석) ===\n")

# 새로운 환자 5명 생성 (마찬가지로 임의의 특징 값)
new_patients_data <- matrix(rnorm(5 * n_features) * c(1, 1.5, 0.5, 1, 1), 
                            nrow = 5, ncol = n_features)
colnames(new_patients_data) <- feature_names

# ★★★ 중요: 새로운 환자 데이터에도 반드시 훈련 세트에서 fit된 preproc를 사용하여 transform해야 함 ★★★
new_patients_scaled <- predict(preproc, as.data.frame(new_patients_data))
dnew <- xgb.DMatrix(data = new_patients_scaled)

# 예측
new_predictions <- predict(model_tuned, dnew)

# SHAP values for new patients (새로운 환자에 대한 SHAP 값 계산)
new_shap_result <- shap.values(model_tuned, new_patients_scaled)
new_shap_vals <- new_shap_result$shap_values

# 결과 출력
cat("\nNew Patient Predictions & Individual Feature Contribution:\n")
for (i in 1:nrow(new_patients_data)) {
  cat(sprintf("\nPatient %d:\n", i))
  cat(sprintf("  Predicted Response Score: %.2f / 100\n", new_predictions[i]))
  
  # 각 환자에 대한 top 3 contributing features
  patient_shap <- new_shap_vals[i, ]
  # 기여도가 큰 순서로 인덱스 추출 (절대값 기준)
  top_features_idx <- order(abs(patient_shap), decreasing = TRUE)[1:3]
  
  cat("  Top 3 Contributing Features (가장 기여도가 높은 특성):\n")
  for (idx in top_features_idx) {
    original_value <- new_patients_data[i, idx]
    shap_value <- patient_shap[idx]
    
    effect <- ifelse(shap_value > 0, "Positive", "Negative")
    
    cat(sprintf("    %s: Input Value=%.2f, SHAP value=%.3f (%s effect)\n",
                feature_names[idx], original_value, shap_value, effect))
  }
}

# 최종 결과 요약
results_df_summary <- data.frame(
  Patient_ID = paste0('Patient_', 1:nrow(new_patients_data)),
  Predicted_Response_Score = round(new_predictions, 2)
)

cat("\n=== Summary of New Predictions (새 환자 예측 요약) ===\n")
print(results_df_summary)

# ============================================
# 9. 최종 요약
# ============================================
cat("\n=== Final Summary (최종 분석 요약) ===\n")
cat("Model: XGBoost Regressor (Tuned)\n")
cat("Best parameters (최적 파라미터):\n")
print(model_tuned_caret$bestTune)
cat(sprintf("\nTuned Model - Test R²: %.4f, Test RMSE: %.4f\n", test_r2_tuned, test_rmse_tuned))
cat("\nTop 3 Most Important Features (가장 중요한 특성 3개):\n")
print(head(importance_matrix, 3))
cat("\n---\n")
cat("분석 결론: 데이터 전처리(Scale/Center)를 적용하고 정규화 파라미터를 포함하여 XGBoost 모델을 Caret으로 튜닝했습니다. SHAP 분석(shapviz)을 통해 모델이 예측에 사용하는 핵심 특성과 해당 특성 값이 예측에 미치는 영향을 깊이 있게 해석할 수 있습니다.\n")
