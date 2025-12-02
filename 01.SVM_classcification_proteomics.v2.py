import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns

# 프로테오믹스 데이터 생성
np.random.seed(42)
n_samples_per_class = 100
n_proteins = 1000

# 4개 질병 타입: Healthy, Disease_A, Disease_B, Disease_C
healthy = np.random.randn(n_samples_per_class, n_proteins) + np.random.uniform(-0.5, 0.5, n_proteins)
disease_a = np.random.randn(n_samples_per_class, n_proteins) + np.random.uniform(0.5, 1.0, n_proteins)
disease_b = np.random.randn(n_samples_per_class, n_proteins) + np.random.uniform(-1.0, -0.5, n_proteins)
disease_c = np.random.randn(n_samples_per_class, n_proteins) + np.random.uniform(1.0, 1.5, n_proteins)

X = np.vstack([healthy, disease_a, disease_b, disease_c])
y = np.array(['Healthy']*n_samples_per_class + 
             ['Disease_A']*n_samples_per_class + 
             ['Disease_B']*n_samples_per_class + 
             ['Disease_C']*n_samples_per_class)

# 데이터프레임 생성
protein_names = [f'Protein_{i+1}' for i in range(n_proteins)]
df = pd.DataFrame(X, columns=protein_names)
df['Label'] = y

print("Data Overview:")
print(df.head())
print(f"\nData shape: {df.shape}")
print(f"\nClass distribution:\n{df['Label'].value_counts()}")

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nLabel mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# Cross-validation (튜닝 전 baseline 확인)
# ============================================
print("\n=== Cross-Validation (Before Tuning) ===")
baseline_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(baseline_svm, X_train_scaled, y_train, cv=cv, scoring='accuracy')

print(f"CV Accuracy scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================
# GridSearchCV로 하이퍼파라미터 튜닝
# ============================================
print("\n=== Hyperparameter Tuning with GridSearchCV ===")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("Starting grid search...")
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# 최적 모델
best_model = grid_search.best_estimator_

# ============================================
# Cross-validation으로 최적 모델 robustness 확인
# ============================================
print("\n=== Cross-Validation (After Tuning) ===")
tuned_cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

print(f"CV Accuracy scores: {tuned_cv_scores}")
print(f"Mean CV Accuracy: {tuned_cv_scores.mean():.4f} (+/- {tuned_cv_scores.std() * 2:.4f})")

# CV 결과 비교 시각화
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(['Baseline', 'Tuned'], [cv_scores.mean(), tuned_cv_scores.mean()], 
        yerr=[cv_scores.std() * 2, tuned_cv_scores.std() * 2], 
        capsize=10, color=['lightblue', 'lightcoral'])
plt.ylabel('Accuracy')
plt.title('Cross-Validation: Baseline vs Tuned')
plt.ylim([0, 1])

plt.subplot(1, 2, 2)
plt.boxplot([cv_scores, tuned_cv_scores], labels=['Baseline', 'Tuned'])
plt.ylabel('Accuracy')
plt.title('CV Score Distribution')
plt.ylim([0, 1])

plt.tight_layout()
plt.show()

# ============================================
# 테스트 세트 성능 평가
# ============================================
print("\n=== Model Performance on Test Set ===")
y_pred = best_model.predict(X_test_scaled)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix - Tuned Multi-class SVM')
plt.tight_layout()
plt.show()

# ============================================
# 새로운 데이터로 예측하기
# ============================================
print("\n\n=== Predicting on New Samples ===")

# 새로운 샘플 5개 생성
new_sample_1 = np.random.randn(1, n_proteins) + np.random.uniform(-0.5, 0.5, n_proteins)
new_sample_2 = np.random.randn(1, n_proteins) + np.random.uniform(0.5, 1.0, n_proteins)
new_sample_3 = np.random.randn(1, n_proteins) + np.random.uniform(-1.0, -0.5, n_proteins)
new_sample_4 = np.random.randn(1, n_proteins) + np.random.uniform(1.0, 1.5, n_proteins)
new_sample_5 = np.random.randn(1, n_proteins)

new_samples = np.vstack([new_sample_1, new_sample_2, new_sample_3, new_sample_4, new_sample_5])

# 스케일링
new_samples_scaled = scaler.transform(new_samples)

# 예측
new_predictions = best_model.predict(new_samples_scaled)
new_predictions_labels = label_encoder.inverse_transform(new_predictions)

# Decision scores
decision_scores = best_model.decision_function(new_samples_scaled)

# 결과 출력
print("\nNew Sample Predictions:")
for i in range(len(new_samples)):
    print(f"\nSample {i+1}:")
    print(f"  Predicted Class: {new_predictions_labels[i]}")
    print(f"  Decision Scores: {decision_scores[i]}")
    print(f"  Most confident class: {label_encoder.classes_[np.argmax(decision_scores[i])]}")

# 결과 정리
results_df = pd.DataFrame({
    'Sample_ID': [f'NewSample_{i+1}' for i in range(len(new_samples))],
    'Predicted_Class': new_predictions_labels,
    'Confidence_Score': [np.max(scores) for scores in decision_scores]
})

print("\n=== Summary of New Predictions ===")
print(results_df)

# GridSearch 결과 상세 정보
print("\n=== GridSearch Top 5 Results ===")
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results_sorted = cv_results.sort_values('rank_test_score')
print(cv_results_sorted[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].head())
