import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

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
             'Disease_A']*n_samples_per_class + 
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
# Linear vs RBF Kernel 비교
# ============================================
print("\n=== Comparing Linear vs RBF Kernel ===")

# Linear kernel 튜닝
print("\nTuning Linear Kernel...")
param_grid_linear = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear']
}

grid_linear = GridSearchCV(
    SVC(random_state=42, probability=True),
    param_grid_linear,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_linear.fit(X_train_scaled, y_train)

print(f"Best Linear params: {grid_linear.best_params_}")
print(f"Best Linear CV accuracy: {grid_linear.best_score_:.4f}")

# RBF kernel 튜닝
print("\nTuning RBF Kernel...")
param_grid_rbf = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

grid_rbf = GridSearchCV(
    SVC(random_state=42, probability=True),
    param_grid_rbf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_rbf.fit(X_train_scaled, y_train)

print(f"Best RBF params: {grid_rbf.best_params_}")
print(f"Best RBF CV accuracy: {grid_rbf.best_score_:.4f}")

# 최적 모델 선택
best_linear = grid_linear.best_estimator_
best_rbf = grid_rbf.best_estimator_

# ============================================
# 테스트 세트 성능 비교 (Accuracy & AUC)
# ============================================
print("\n=== Test Set Performance Comparison ===")

# Linear kernel 성능
y_pred_linear = best_linear.predict(X_test_scaled)
y_prob_linear = best_linear.predict_proba(X_test_scaled)
acc_linear = accuracy_score(y_test, y_pred_linear)

# RBF kernel 성능
y_pred_rbf = best_rbf.predict(X_test_scaled)
y_prob_rbf = best_rbf.predict_proba(X_test_scaled)
acc_rbf = accuracy_score(y_test, y_pred_rbf)

print(f"\nLinear Kernel Test Accuracy: {acc_linear:.4f}")
print(f"RBF Kernel Test Accuracy: {acc_rbf:.4f}")

# Multi-class AUC 계산 (One-vs-Rest)
y_test_binarized = label_binarize(y_test, classes=range(len(label_encoder.classes_)))

# Linear AUC
auc_linear = roc_auc_score(y_test_binarized, y_prob_linear, average='macro', multi_class='ovr')
print(f"\nLinear Kernel AUC (macro): {auc_linear:.4f}")

# RBF AUC
auc_rbf = roc_auc_score(y_test_binarized, y_prob_rbf, average='macro', multi_class='ovr')
print(f"RBF Kernel AUC (macro): {auc_rbf:.4f}")

# 성능 비교 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy & AUC 비교
metrics = ['Accuracy', 'AUC']
linear_scores = [acc_linear, auc_linear]
rbf_scores = [acc_rbf, auc_rbf]

x = np.arange(len(metrics))
width = 0.35

axes[0].bar(x - width/2, linear_scores, width, label='Linear', color='skyblue')
axes[0].bar(x + width/2, rbf_scores, width, label='RBF', color='lightcoral')
axes[0].set_ylabel('Score')
axes[0].set_title('Linear vs RBF Kernel Performance')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].legend()
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)

# Confusion Matrix 비교 (RBF만 표시)
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_,
            ax=axes[1])
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')
axes[1].set_title('Confusion Matrix - RBF Kernel')

plt.tight_layout()
plt.show()

# Class-wise AUC 출력
print("\n=== Per-Class AUC (RBF Kernel) ===")
for i, class_name in enumerate(label_encoder.classes_):
    class_auc = roc_auc_score(y_test_binarized[:, i], y_prob_rbf[:, i])
    print(f"{class_name}: {class_auc:.4f}")

# ============================================
# Learning Curve 분석
# ============================================
print("\n=== Learning Curve Analysis ===")

# 더 나은 성능의 모델로 learning curve 분석 (RBF 사용)
best_model = best_rbf if acc_rbf >= acc_linear else best_linear
kernel_used = 'RBF' if acc_rbf >= acc_linear else 'Linear'

print(f"Using {kernel_used} kernel for learning curve analysis...")

train_sizes = np.linspace(0.1, 1.0, 10)

train_sizes_abs, train_scores, val_scores = learning_curve(
    best_model,
    X_train_scaled,
    y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# 평균 및 표준편차 계산
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

print(f"\nTraining sizes: {train_sizes_abs}")
print(f"Train scores (mean): {train_mean}")
print(f"Validation scores (mean): {val_mean}")

# Learning Curve 시각화
plt.figure(figsize=(10, 6))
plt.plot(train_sizes_abs, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes_abs, val_mean, 'o-', color='g', label='Cross-validation score')

plt.fill_between(train_sizes_abs, 
                 train_mean - train_std, 
                 train_mean + train_std, 
                 alpha=0.1, color='r')
plt.fill_between(train_sizes_abs, 
                 val_mean - val_std, 
                 val_mean + val_std, 
                 alpha=0.1, color='g')

plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title(f'Learning Curve - {kernel_used} Kernel SVM')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Learning curve 분석 결과 해석
print("\n=== Learning Curve Interpretation ===")
final_gap = train_mean[-1] - val_mean[-1]
print(f"Final training score: {train_mean[-1]:.4f}")
print(f"Final validation score: {val_mean[-1]:.4f}")
print(f"Train-Val gap: {final_gap:.4f}")

if final_gap > 0.1:
    print("→ High variance (overfitting) detected. Consider: more data, regularization, or simpler model")
elif val_mean[-1] < 0.7:
    print("→ High bias (underfitting) detected. Consider: more features, complex model, or less regularization")
else:
    print("→ Model shows good balance between bias and variance")

# ============================================
# ROC Curve (Multi-class, RBF kernel만)
# ============================================
print("\n=== ROC Curves (RBF Kernel) ===")

n_classes = len(label_encoder.classes_)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob_rbf[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'red', 'green', 'orange'])

for i, color, class_name in zip(range(n_classes), colors, label_encoder.classes_):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_name} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Multi-class Classification (RBF Kernel)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================
# 새로운 데이터로 예측
# ============================================
print("\n=== Predicting on New Samples ===")

new_sample_1 = np.random.randn(1, n_proteins) + np.random.uniform(-0.5, 0.5, n_proteins)
new_sample_2 = np.random.randn(1, n_proteins) + np.random.uniform(0.5, 1.0, n_proteins)
new_sample_3 = np.random.randn(1, n_proteins) + np.random.uniform(-1.0, -0.5, n_proteins)
new_sample_4 = np.random.randn(1, n_proteins) + np.random.uniform(1.0, 1.5, n_proteins)
new_sample_5 = np.random.randn(1, n_proteins)

new_samples = np.vstack([new_sample_1, new_sample_2, new_sample_3, new_sample_4, new_sample_5])
new_samples_scaled = scaler.transform(new_samples)

# 예측 (best model 사용)
new_predictions = best_model.predict(new_samples_scaled)
new_predictions_labels = label_encoder.inverse_transform(new_predictions)
new_probabilities = best_model.predict_proba(new_samples_scaled)

# 결과 출력
print(f"\nPredictions using {kernel_used} kernel:")
for i in range(len(new_samples)):
    print(f"\nSample {i+1}:")
    print(f"  Predicted Class: {new_predictions_labels[i]}")
    print(f"  Probabilities: {dict(zip(label_encoder.classes_, new_probabilities[i]))}")
    print(f"  Confidence: {np.max(new_probabilities[i]):.4f}")

# 결과 정리
results_df = pd.DataFrame({
    'Sample_ID': [f'NewSample_{i+1}' for i in range(len(new_samples))],
    'Predicted_Class': new_predictions_labels,
    'Confidence': [np.max(prob) for prob in new_probabilities]
})

print("\n=== Summary of New Predictions ===")
print(results_df)

# 최종 요약
print("\n=== Final Summary ===")
print(f"Best performing kernel: {kernel_used}")
print(f"Test Accuracy: {max(acc_linear, acc_rbf):.4f}")
print(f"Test AUC (macro): {max(auc_linear, auc_rbf):.4f}")
print(f"Best hyperparameters: {best_model.get_params()}")
