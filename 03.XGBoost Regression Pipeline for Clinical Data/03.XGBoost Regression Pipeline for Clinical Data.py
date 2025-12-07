import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time # 실행 시간 측정을 위해 추가

# Matplotlib 및 Seaborn 설정 (한글 폰트 문제 해결은 제외하고, 기본 설정만 유지)
plt.style.use('ggplot')

# ============================================
# 임상 데이터 생성 및 비선형성 부여
# ============================================
np.random.seed(42)

n_samples = 500
n_features = 30  # 임상 지표 (나이, 혈압, 혈당, 바이오마커 등)

# 임상 특징 생성 (평균 0, 표준편차 1)
X = np.random.randn(n_samples, n_features)

# 실제 환경 시뮬레이션: 타겟 변수에 복잡한 관계 부여
true_weights = np.random.randn(n_features) * 2

# 1. 상호작용 항 (Interaction Term): Feature 1과 Feature 6의 곱
interaction_term = X[:, 0] * X[:, 5] * 5

# 2. 비선형 항 (Nonlinear Term): Feature 11의 지수 함수
nonlinear_term = np.exp(X[:, 10] / 3) * 4

# 3. 중요하지 않은 특징의 가중치
noise_features_term = X[:, 20:30].sum(axis=1) * 0.5 

# 치료 반응 점수 생성 (0-100 사이, 복잡한 특징 관계 포함)
y = (X @ true_weights * 2 + interaction_term + nonlinear_term + noise_features_term +
      np.random.randn(n_samples) * 8) # 노이즈 레벨 증가

# 0-100 범위로 스케일링 (Response Score)
y = (y - y.min()) / (y.max() - y.min()) * 100

# 데이터프레임 생성
feature_names = [f'Clinical_Feature_{i+1}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['Response_Score'] = y

print("Data Overview (데이터 개요):")
print(df.head())
print(f"\nData shape (데이터 형태): {df.shape}")
print(f"\nResponse Score statistics (반응 점수 통계):")
print(df['Response_Score'].describe())

# ============================================
# Train-test split & Data Preprocessing (데이터 전처리)
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 실제 임상 데이터처럼 스케일링이 필요하다고 가정하고 StandardScaler 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain set (훈련 세트): {X_train_scaled.shape}, Test set (테스트 세트): {X_test_scaled.shape}")
print("Data has been scaled (데이터 스케일링 완료).")

# ============================================
# Hyperparameter Tuning with GridSearchCV (하이퍼파라미터 튜닝)
# ============================================
print("\n=== Hyperparameter Tuning (하이퍼파라미터 튜닝 시작) ===")
start_time = time.time()

# 정규화 파라미터 (reg_alpha, reg_lambda)를 추가하여 탐색 범위 확장
param_grid = {
    'n_estimators': [150, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'reg_alpha': [0, 0.1, 0.5], # L1 정규화 추가
    'reg_lambda': [0.1, 1, 5]   # L2 정규화 범위 조정
}

# XGBoost 모델 설정
xgb_model = xgb.XGBRegressor(
    random_state=42, 
    n_jobs=-1, 
    objective='reg:squarederror' # 회귀 분석 명시
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5, # 5-Fold Cross-Validation
    scoring='neg_mean_squared_error',
    verbose=0, # 출력 간소화
    n_jobs=-1
)

# 스케일된 훈련 데이터로 튜닝 수행
grid_search.fit(X_train_scaled, y_train) 

end_time = time.time()

print(f"\nGrid Search Completed in {end_time - start_time:.2f} seconds.")
print("Best parameters (최적 파라미터):")
print(grid_search.best_params_)
print(f"Best CV score (최적 CV 점수, neg_MSE): {grid_search.best_score_:.4f}")

# 최적 모델
model_tuned = grid_search.best_estimator_

# 예측
y_pred_train_tuned = model_tuned.predict(X_train_scaled)
y_pred_test_tuned = model_tuned.predict(X_test_scaled)

# 성능 평가
test_mse_tuned = mean_squared_error(y_test, y_pred_test_tuned)
test_rmse_tuned = np.sqrt(test_mse_tuned)
test_mae_tuned = mean_absolute_error(y_test, y_pred_test_tuned)
test_r2_tuned = r2_score(y_test, y_pred_test_tuned)

print("\n=== Tuned Model Test Performance (최적 모델 테스트 성능) ===")
print(f"Test MSE: {test_mse_tuned:.4f}")
print(f"Test RMSE: {test_rmse_tuned:.4f}")
print(f"Test MAE: {test_mae_tuned:.4f}")
print(f"Test R²: {test_r2_tuned:.4f}")

# ============================================
# Feature Importance 분석 및 시각화
# ============================================
print("\n=== Feature Importance Analysis (특성 중요도 분석) ===")

# XGBoost 내장 feature importance
feature_importance = model_tuned.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features (가장 중요한 특성 10개):")
print(feature_importance_df.head(10))

# Feature importance 시각화
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'].head(15), 
         feature_importance_df['Importance'].head(15), color='darkred')
plt.xlabel('Importance (중요도)')
plt.title('Top 15 Feature Importances (상위 15개 특성 중요도)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================
# SHAP 분석 (모델 해석력 강화)
# ============================================
print("\n=== SHAP Analysis (SHAP 분석) ===")

# SHAP explainer 생성
explainer = shap.TreeExplainer(model_tuned)
# 스케일된 테스트 데이터로 SHAP 값 계산
shap_values = explainer.shap_values(X_test_scaled) 

# 1. SHAP summary plot (전역적 특성 중요도 및 영향 방향)
print("\nSHAP Summary Plot: Global Feature Influence")
plt.figure(figsize=(10, 6))
# SHAP 플롯은 스케일된 데이터 X_test_scaled를 사용하지만, feature_names를 전달하여 레이블링합니다.
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.show()

# 2. SHAP dependence plot (상위 1개 특성의 영향 관계 시각화)
top_feature_index = feature_importance_df.index[0]
top_feature_name = feature_names[top_feature_index]

print(f"\nSHAP Dependence Plot for Top Feature: {top_feature_name}")
plt.figure(figsize=(8, 6))
# SHAP dependence_plot은 특성 값(X_test_scaled)과 SHAP 값(shap_values)을 사용
shap.dependence_plot(
    top_feature_name, 
    shap_values, 
    X_test_scaled, 
    feature_names=feature_names, 
    interaction_index=None, # 상호작용은 제외하고 주 효과만 시각화
    show=False
)
plt.title(f'SHAP Dependence Plot for {top_feature_name}')
plt.tight_layout()
plt.show()

# ============================================
# 예측 vs 실제 시각화
# ============================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_test_tuned, alpha=0.6, color='b')
# 완벽한 예측 선 (y=x)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('True Response Score (실제 반응 점수)')
plt.ylabel('Predicted Response Score (예측 반응 점수)')
plt.title(f'Predictions vs True Values (예측 대 실제, R²={test_r2_tuned:.3f})')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
residuals = y_test - y_pred_test_tuned
plt.scatter(y_pred_test_tuned, residuals, alpha=0.6, color='g')
plt.axhline(y=0, color='r', linestyle='-', lw=2)
plt.xlabel('Predicted Response Score (예측 반응 점수)')
plt.ylabel('Residuals (잔차)')
plt.title('Residual Plot (잔차 플롯)')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 새로운 환자 데이터로 예측 및 개별 해석
# ============================================
print("\n=== Predicting on New Patients (새로운 환자 예측 및 개별 해석) ===")

# 새로운 환자 5명 생성 (마찬가지로 임의의 특징 값)
new_patients = np.random.randn(5, n_features) * [1, 1.5, 0.5, 1, 1]

# ★★★ 중요: 새로운 환자 데이터에도 반드시 훈련 세트에서 fit된 scaler를 사용하여 transform해야 함 ★★★
new_patients_scaled = scaler.transform(new_patients)

# 예측
new_predictions = model_tuned.predict(new_patients_scaled)

# SHAP values for new patients (새로운 환자에 대한 SHAP 값 계산)
new_shap_values = explainer.shap_values(new_patients_scaled)

# 결과 출력
print("\nNew Patient Predictions & Individual Feature Contribution:")
for i in range(len(new_patients)):
    print(f"\nPatient {i+1}:")
    print(f"  Predicted Response Score: {new_predictions[i]:.2f} / 100")
    
    # 각 환자에 대한 top 3 contributing features
    patient_shap = new_shap_values[i]
    # 기여도가 큰 순서로 인덱스 추출 (절대값 기준)
    top_features_idx = np.argsort(np.abs(patient_shap))[-3:][::-1]
    
    print(f"  Top 3 Contributing Features (가장 기여도가 높은 특성):")
    for idx in top_features_idx:
        # 스케일링 전의 실제 입력 값도 출력하여 해석에 도움을 줍니다.
        original_value = new_patients[i, idx] 
        shap_value = patient_shap[idx]
        
        print(f"    {feature_names[idx]}: Input Value={original_value:.2f}, SHAP value={shap_value:.3f} ({'Positive' if shap_value > 0 else 'Negative'} effect)")

# 최종 결과 요약
results_df = pd.DataFrame({
    'Patient_ID': [f'Patient_{i+1}' for i in range(len(new_patients))],
    'Predicted_Response_Score': np.round(new_predictions, 2)
})

print("\n=== Summary of New Predictions (새 환자 예측 요약) ===")
print(results_df)

# ============================================
# 최종 요약
# ============================================
print("\n=== Final Summary (최종 분석 요약) ===")
print(f"Model: XGBoost Regressor (Tuned)")
print(f"Best parameters (최적 파라미터):")
print(f"  {model_tuned.get_params()}")
print(f"\nTuned Model - Test R²: {test_r2_tuned:.4f}, Test RMSE: {test_rmse_tuned:.4f}")
print(f"\nTop 3 Most Important Features (가장 중요한 특성 3개):")
for i in range(3):
    print(f"  {i+1}. {feature_importance_df.iloc[i]['Feature']}: "
          f"Importance={feature_importance_df.iloc[i]['Importance']:.4f}") 
print("\n---")
print("분석 결론: 데이터 전처리(StandardScaler)를 적용하고 정규화 파라미터를 추가하여 모델의 일반화 성능을 높였습니다. SHAP 분석을 통해 모델이 예측에 사용하는 핵심 특성(예: Feature_1과 Feature_6의 상호작용)과 해당 특성 값이 예측에 미치는 영향을 깊이 있게 해석할 수 있습니다.")
