"""
Feature Selection & Model Interpretability Pipeline
===================================================

목적: 바이오인포매틱스 데이터에서 feature selection과 모델 해석 연습

학습 목표:
1. L1/L2 정규화 비교
2. XGBoost feature importance
3. Permutation importance
4. SHAP values로 모델 해석

데이터: Synthetic gene expression (100 samples, 500 genes)
태스크: Binary classification (cancer subtype)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap

# 재현성
np.random.seed(123)

#%%
# =============================================================================
# 1단계: 데이터 생성
# =============================================================================

n_samples = 100
n_genes = 500
important_genes = list(range(20))  # Gene_0 ~ Gene_19가 실제 중요

# Gene expression matrix
X = np.random.randn(n_samples, n_genes)
gene_names = [f"Gene_{i}" for i in range(n_genes)]

# Target: 중요한 유전자 20개의 합으로 결정
y = (X[:, important_genes].sum(axis=1) > 0).astype(int)

# DataFrame 생성
X_df = pd.DataFrame(X, columns=gene_names)
y_series = pd.Series(y, name="Subtype")

print(f"데이터 shape: {X_df.shape}")
print(f"Class distribution: {np.bincount(y)}")

#%%
# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_series, test_size=0.3, random_state=123, stratify=y_series
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

#%%
# =============================================================================
# 2단계: L1 정규화 (Lasso)
# =============================================================================

# Lasso: alpha=1 (L1), C는 정규화 강도의 역수
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=123)
model_l1.fit(X_train, y_train)

# 선택된 feature (coefficient != 0)
coef_l1 = pd.Series(model_l1.coef_[0], index=gene_names)
selected_l1 = coef_l1[coef_l1 != 0].index.tolist()

# 성능
pred_l1 = model_l1.predict(X_test)
acc_l1 = accuracy_score(y_test, pred_l1)

print(f"\n=== L1 (Lasso) ===")
print(f"Selected features: {len(selected_l1)}")
print(f"Accuracy: {acc_l1:.3f}")

#%%
# =============================================================================
# 3단계: L2 정규화 (Ridge)
# =============================================================================

model_l2 = LogisticRegression(penalty='l2', solver='liblinear', C=0.1, random_state=123)
model_l2.fit(X_train, y_train)

coef_l2 = pd.Series(model_l2.coef_[0], index=gene_names)

# 성능
pred_l2 = model_l2.predict(X_test)
acc_l2 = accuracy_score(y_test, pred_l2)

print(f"\n=== L2 (Ridge) ===")
print(f"Non-zero coefficients: {np.sum(coef_l2 != 0)}")  # 모두 non-zero
print(f"Accuracy: {acc_l2:.3f}")

#%%
# =============================================================================
# 4단계: XGBoost
# =============================================================================

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'eta': 0.1,
    'eval_metric': 'logloss'
}

evals = [(dtrain, 'train'), (dtest, 'test')]
model_xgb = xgb.train(
    params, 
    dtrain, 
    num_boost_round=100, 
    evals=evals, 
    verbose_eval=False
)

# 예측
pred_xgb_prob = model_xgb.predict(dtest)
pred_xgb = (pred_xgb_prob > 0.5).astype(int)
acc_xgb = accuracy_score(y_test, pred_xgb)

print(f"\n=== XGBoost ===")
print(f"Accuracy: {acc_xgb:.3f}")

#%%
# Built-in feature importance
importance_gain = model_xgb.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': list(importance_gain.keys()),
    'Gain': list(importance_gain.values())
}).sort_values('Gain', ascending=False)

# Feature 이름 매핑 (f0 -> Gene_0)
feature_map = {f'f{i}': gene_names[i] for i in range(n_genes)}
importance_df['Feature'] = importance_df['Feature'].map(feature_map)

print("\nTop 10 features by Gain:")
print(importance_df.head(10))

#%%
# =============================================================================
# 5단계: Permutation Importance
# =============================================================================

# scikit-learn의 permutation_importance 사용
# XGBoost를 sklearn wrapper로 변환
from xgboost import XGBClassifier

model_xgb_sk = XGBClassifier(
    max_depth=3, 
    learning_rate=0.1, 
    n_estimators=100, 
    random_state=123
)
model_xgb_sk.fit(X_train, y_train)

# Permutation importance 계산
perm_result = permutation_importance(
    model_xgb_sk, 
    X_test, 
    y_test, 
    n_repeats=10, 
    random_state=123
)

perm_importance_df = pd.DataFrame({
    'Feature': gene_names,
    'Importance': perm_result.importances_mean
}).sort_values('Importance', ascending=False)

print("\n=== Permutation Importance ===")
print("Top 10 features:")
print(perm_importance_df.head(10))

#%%
# =============================================================================
# 6단계: SHAP Values
# =============================================================================

# SHAP explainer 생성 (Tree explainer for XGBoost)
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test)

print("\n=== SHAP Values ===")
print(f"SHAP values shape: {shap_values.shape}")

#%%
# SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_summary_bar.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# SHAP beeswarm plot (상세)
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# 개별 샘플 SHAP waterfall plot
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0], 
    base_values=explainer.expected_value, 
    data=X_test.iloc[0],
    feature_names=gene_names
), show=False)
plt.tight_layout()
plt.savefig('shap_waterfall_sample0.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# 7단계: 결과 비교 시각화
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) L1 Coefficients
top_l1 = coef_l1.abs().sort_values(ascending=False).head(20)
axes[0, 0].barh(range(len(top_l1)), top_l1.values)
axes[0, 0].set_yticks(range(len(top_l1)))
axes[0, 0].set_yticklabels(top_l1.index, fontsize=8)
axes[0, 0].set_xlabel('Abs(Coefficient)')
axes[0, 0].set_title('L1 (Lasso) Top 20 Features')
axes[0, 0].invert_yaxis()

# 2) XGBoost Feature Importance (Gain)
top_gain = importance_df.head(20)
axes[0, 1].barh(range(len(top_gain)), top_gain['Gain'].values)
axes[0, 1].set_yticks(range(len(top_gain)))
axes[0, 1].set_yticklabels(top_gain['Feature'], fontsize=8)
axes[0, 1].set_xlabel('Gain')
axes[0, 1].set_title('XGBoost Feature Importance (Gain)')
axes[0, 1].invert_yaxis()

# 3) Permutation Importance
top_perm = perm_importance_df.head(20)
axes[1, 0].barh(range(len(top_perm)), top_perm['Importance'].values)
axes[1, 0].set_yticks(range(len(top_perm)))
axes[1, 0].set_yticklabels(top_perm['Feature'], fontsize=8)
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Permutation Importance')
axes[1, 0].invert_yaxis()

# 4) SHAP Mean Absolute Values
shap_importance = pd.DataFrame({
    'Feature': gene_names,
    'SHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('SHAP', ascending=False)

top_shap = shap_importance.head(20)
axes[1, 1].barh(range(len(top_shap)), top_shap['SHAP'].values)
axes[1, 1].set_yticks(range(len(top_shap)))
axes[1, 1].set_yticklabels(top_shap['Feature'], fontsize=8)
axes[1, 1].set_xlabel('Mean |SHAP value|')
axes[1, 1].set_title('SHAP Feature Importance')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# 8단계: Ground Truth 비교
# =============================================================================

true_important = [f"Gene_{i}" for i in range(20)]

# 각 방법이 찾은 실제 중요 유전자
detected_l1 = set(selected_l1) & set(true_important)
detected_gain = set(importance_df.head(20)['Feature']) & set(true_important)
detected_perm = set(perm_importance_df.head(20)['Feature']) & set(true_important)
detected_shap = set(shap_importance.head(20)['Feature']) & set(true_important)

print("\n=== Ground Truth Comparison ===")
print(f"L1 detected: {len(detected_l1)} / 20")
print(f"XGBoost Gain detected: {len(detected_gain)} / 20")
print(f"Permutation detected: {len(detected_perm)} / 20")
print(f"SHAP detected: {len(detected_shap)} / 20")

#%%
# Venn diagram style 비교
comparison_df = pd.DataFrame({
    'Method': ['L1', 'XGBoost Gain', 'Permutation', 'SHAP'],
    'Detected': [len(detected_l1), len(detected_gain), 
                 len(detected_perm), len(detected_shap)],
    'Accuracy': [acc_l1, acc_xgb, acc_xgb, acc_xgb]
})

print("\n=== Summary ===")
print(comparison_df)

#%%
# =============================================================================
# 9단계: 저장
# =============================================================================

# 결과 저장
results = {
    'l1_coefficients': coef_l1,
    'l2_coefficients': coef_l2,
    'xgb_importance': importance_df,
    'permutation_importance': perm_importance_df,
    'shap_importance': shap_importance,
    'comparison': comparison_df
}

# CSV로 저장
for key, df in results.items():
    if isinstance(df, pd.DataFrame):
        df.to_csv(f'{key}.csv', index=False)
    elif isinstance(df, pd.Series):
        df.to_csv(f'{key}.csv')

print("\n결과 저장 완료")
print("생성된 파일:")
print("- feature_importance_comparison.png")
print("- shap_summary_bar.png")
print("- shap_summary_beeswarm.png")
print("- shap_waterfall_sample0.png")
print("- *.csv (각 결과)")
