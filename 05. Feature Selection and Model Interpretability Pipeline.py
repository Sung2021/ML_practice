"""
Feature Selection & Model Interpretability Pipeline (Refactored)
목적: 바이오인포매틱스 데이터에서 feature selection과 모델 해석 연습
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
from xgboost import XGBClassifier
import shap
import warnings

# 경고 무시 (주로 liblinear solver warning)
warnings.filterwarnings('ignore')
np.random.seed(123)

# =============================================================================
# 0. 설정 및 데이터 생성 함수
# =============================================================================

def create_synthetic_data(n_samples=100, n_genes=500, n_important=20):
    """합성 유전자 발현 데이터 및 타겟 생성"""
    
    # Feature names
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    important_genes_idx = list(range(n_important))

    # Gene expression matrix (X)
    X = np.random.randn(n_samples, n_genes)

    # Target (Y): 중요한 유전자들의 합으로 결정
    y = (X[:, important_genes_idx].sum(axis=1) > 0).astype(int)

    # DataFrame 및 Series 생성
    X_df = pd.DataFrame(X, columns=gene_names)
    y_series = pd.Series(y, name="Subtype")
    
    print(f"데이터 shape: {X_df.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X_df, y_series, gene_names, [f"Gene_{i}" for i in important_genes_idx]

# =============================================================================
# 1. 모델 학습 및 L1/L2 비교
# =============================================================================

def train_and_compare_linear_models(X_train, X_test, y_train, y_test, gene_names):
    """L1 (Lasso) 및 L2 (Ridge) 로지스틱 회귀 모델 학습 및 비교"""
    
    results = {}

    # L1 (Lasso)
    model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=123)
    model_l1.fit(X_train, y_train)
    coef_l1 = pd.Series(model_l1.coef_[0], index=gene_names)
    selected_l1 = coef_l1[coef_l1 != 0].index.tolist()
    acc_l1 = accuracy_score(y_test, model_l1.predict(X_test))
    
    print("\n=== L1 (Lasso) ===")
    print(f"Selected features: {len(selected_l1)}")
    print(f"Accuracy: {acc_l1:.3f}")

    # L2 (Ridge)
    model_l2 = LogisticRegression(penalty='l2', solver='liblinear', C=0.1, random_state=123)
    model_l2.fit(X_train, y_train)
    coef_l2 = pd.Series(model_l2.coef_[0], index=gene_names)
    acc_l2 = accuracy_score(y_test, model_l2.predict(X_test))

    print("\n=== L2 (Ridge) ===")
    print(f"Accuracy: {acc_l2:.3f}")
    
    results['l1_coef'] = coef_l1
    results['selected_l1'] = selected_l1
    results['acc_l1'] = acc_l1
    results['acc_l2'] = acc_l2
    
    return results

def train_xgboost(X_train, X_test, y_train, y_test):
    """XGBoost 모델 학습 및 기본 중요도 계산"""
    
    # 1. XGBoost DMatrix 및 학습 (자세한 중요도 분석을 위해 일반 xgb.train 사용)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'eta': 0.1,
        'eval_metric': 'logloss'
    }

    model_xgb = xgb.train(
        params, 
        dtrain, 
        num_boost_round=100, 
        verbose_eval=False
    )
    
    # 예측 및 성능
    pred_xgb_prob = model_xgb.predict(dtest)
    pred_xgb = (pred_xgb_prob > 0.5).astype(int)
    acc_xgb = accuracy_score(y_test, pred_xgb)

    print(f"\n=== XGBoost ===")
    print(f"Accuracy: {acc_xgb:.3f}")
    
    # 2. Built-in feature importance (Gain)
    importance_gain = model_xgb.get_score(importance_type='gain')
    
    # Feature 이름 매핑 (f0 -> Gene_0)
    feature_map = {f'f{i}': X_train.columns[i] for i in range(X_train.shape[1])}
    
    importance_df = pd.DataFrame({
        'Feature': list(importance_gain.keys()),
        'Gain': list(importance_gain.values())
    }).sort_values('Gain', ascending=False)
    
    importance_df['Feature'] = importance_df['Feature'].map(feature_map)
    
    return model_xgb, importance_df, acc_xgb

# =============================================================================
# 2. 모델 해석: Permutation & SHAP
# =============================================================================

def calculate_permutation_importance(model_xgb, X_test, y_test, gene_names):
    """Permutation Importance 계산"""
    
    # Sklearn wrapper로 변환 후 학습 (permutation_importance를 위해)
    model_xgb_sk = XGBClassifier(
        max_depth=3, 
        learning_rate=0.1, 
        n_estimators=100, 
        random_state=123,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model_xgb_sk.fit(X_test, y_test) # test set에서 계산하지만, 모델은 train set에서 학습된 것으로 가정

    perm_result = permutation_importance(
        model_xgb_sk, 
        X_test, 
        y_test, 
        n_repeats=10, 
        random_state=123,
        scoring='accuracy'
    )

    perm_importance_df = pd.DataFrame({
        'Feature': gene_names,
        'Importance': perm_result.importances_mean
    }).sort_values('Importance', ascending=False)
    
    return perm_importance_df

def calculate_shap_values(model_xgb, X_test, gene_names):
    """SHAP Values 계산"""
    
    explainer = shap.TreeExplainer(model_xgb)
    shap_values = explainer.shap_values(X_test)
    
    # Global SHAP importance (Mean Absolute SHAP)
    shap_importance = pd.DataFrame({
        'Feature': gene_names,
        'SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values('SHAP', ascending=False)
    
    return shap_values, explainer.expected_value, shap_importance

# =============================================================================
# 3. 시각화 함수
# =============================================================================

def plot_importance_comparison(results, perm_df, shap_df, gene_names, output_filename='feature_importance_comparison.png'):
    """다양한 특징 중요도 방법을 비교하는 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1) L1 Coefficients (Abs)
    coef_l1 = results['l1_coef']
    top_l1 = coef_l1.abs().sort_values(ascending=False).head(20)
    axes[0, 0].barh(range(len(top_l1)), top_l1.values)
    axes[0, 0].set_yticks(range(len(top_l1)))
    axes[0, 0].set_yticklabels(top_l1.index, fontsize=8)
    axes[0, 0].set_xlabel('Abs(Coefficient)')
    axes[0, 0].set_title('A. L1 (Lasso) Top 20 Features')
    axes[0, 0].invert_yaxis()

    # 2) XGBoost Feature Importance (Gain)
    importance_df = results['xgb_importance']
    top_gain = importance_df.head(20)
    axes[0, 1].barh(range(len(top_gain)), top_gain['Gain'].values)
    axes[0, 1].set_yticks(range(len(top_gain)))
    axes[0, 1].set_yticklabels(top_gain['Feature'], fontsize=8)
    axes[0, 1].set_xlabel('Gain')
    axes[0, 1].set_title('B. XGBoost Feature Importance (Gain)')
    axes[0, 1].invert_yaxis()

    # 3) Permutation Importance
    top_perm = perm_df.head(20)
    axes[1, 0].barh(range(len(top_perm)), top_perm['Importance'].values)
    axes[1, 0].set_yticks(range(len(top_perm)))
    axes[1, 0].set_yticklabels(top_perm['Feature'], fontsize=8)
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('C. Permutation Importance')
    axes[1, 0].invert_yaxis()

    # 4) SHAP Mean Absolute Values
    top_shap = shap_df.head(20)
    axes[1, 1].barh(range(len(top_shap)), top_shap['SHAP'].values)
    axes[1, 1].set_yticks(range(len(top_shap)))
    axes[1, 1].set_yticklabels(top_shap['Feature'], fontsize=8)
    axes[1, 1].set_xlabel('Mean |SHAP value|')
    axes[1, 1].set_title('D. SHAP Feature Importance')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_shap_results(shap_values, X_test, explainer_value, gene_names):
    """SHAP Summary 및 Waterfall Plot 시각화"""

    # SHAP summary bar plot
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar)")
    plt.tight_layout()
    plt.savefig('shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.show()

    # SHAP beeswarm plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary (Beeswarm)")
    plt.tight_layout()
    plt.savefig('shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 개별 샘플 SHAP waterfall plot (첫 번째 샘플)
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0], 
        base_values=explainer_value, 
        data=X_test.iloc[0],
        feature_names=gene_names
    ), show=False)
    plt.title(f"SHAP Waterfall Plot for Sample 0")
    plt.tight_layout()
    plt.savefig('shap_waterfall_sample0.png', dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# 4. 메인 실행 함수
# =============================================================================

def main():
    
    # 데이터 생성
    X_df, y_series, gene_names, true_important = create_synthetic_data()

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=123, stratify=y_series
    )

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    
    # L1/L2 모델 학습 및 결과
    linear_results = train_and_compare_linear_models(X_train, X_test, y_train, y_test, gene_names)
    
    # XGBoost 모델 학습 및 Gain 중요도
    model_xgb, importance_gain_df, acc_xgb = train_xgboost(X_train, X_test, y_train, y_test)
    linear_results['xgb_importance'] = importance_gain_df
    
    # Permutation Importance
    perm_importance_df = calculate_permutation_importance(model_xgb, X_test, y_test, gene_names)
    
    # SHAP Values
    shap_values, explainer_value, shap_importance_df = calculate_shap_values(model_xgb, X_test, gene_names)

    # =========================================================================
    # 결과 비교 및 시각화
    # =========================================================================
    
    # 특징 중요도 비교 시각화
    plot_importance_comparison(linear_results, perm_importance_df, shap_importance_df, gene_names)
    
    # SHAP 시각화
    plot_shap_results(shap_values, X_test, explainer_value, gene_names)

    # =========================================================================
    # Ground Truth 비교 및 요약
    # =========================================================================

    def compare_to_ground_truth(df, col_name, top_n=20):
        """특정 방법론이 실제 중요 유전자를 얼마나 찾았는지 계산"""
        detected = set(df.head(top_n)['Feature']) & set(true_important)
        return len(detected)

    detected_l1 = len(set(linear_results['selected_l1']) & set(true_important))
    detected_gain = compare_to_ground_truth(importance_gain_df, 'Gain')
    detected_perm = compare_to_ground_truth(perm_importance_df, 'Importance')
    detected_shap = compare_to_ground_truth(shap_importance_df, 'SHAP')

    print("\n=== Ground Truth Comparison (Top 20) ===")
    comparison_df = pd.DataFrame({
        'Method': ['L1', 'XGBoost Gain', 'Permutation', 'SHAP'],
        'Accuracy': [linear_results['acc_l1'], acc_xgb, acc_xgb, acc_xgb],
        'Detected_True_Genes': [detected_l1, detected_gain, detected_perm, detected_shap]
    })
    
    print(comparison_df)

    # 결과 저장
    comparison_df.to_csv('feature_selection_summary.csv', index=False)
    print("\n최종 요약 결과 저장 완료: feature_selection_summary.csv")


if __name__ == '__main__':
    main()
