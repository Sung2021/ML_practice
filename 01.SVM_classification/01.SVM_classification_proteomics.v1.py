import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

# SVM 모델 학습 (RBF kernel for multi-class)
print("\nTraining SVM model...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 테스트 세트 예측
y_pred = svm_model.predict(X_test_scaled)

# 성능 평가
print("\n=== Model Performance on Test Set ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

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
plt.title('Confusion Matrix - Multi-class SVM')
plt.tight_layout()
plt.show()

# ============================================
# 새로운 데이터로 예측하기
# ============================================
print("\n\n=== Predicting on New Samples ===")

# 새로운 샘플 5개 생성 (실제로는 새로운 환자 데이터)
new_sample_1 = np.random.randn(1, n_proteins) + np.random.uniform(-0.5, 0.5, n_proteins)  # Healthy 유사
new_sample_2 = np.random.randn(1, n_proteins) + np.random.uniform(0.5, 1.0, n_proteins)   # Disease_A 유사
new_sample_3 = np.random.randn(1, n_proteins) + np.random.uniform(-1.0, -0.5, n_proteins) # Disease_B 유사
new_sample_4 = np.random.randn(1, n_proteins) + np.random.uniform(1.0, 1.5, n_proteins)   # Disease_C 유사
new_sample_5 = np.random.randn(1, n_proteins)  # 애매한 케이스

new_samples = np.vstack([new_sample_1, new_sample_2, new_sample_3, new_sample_4, new_sample_5])

# 새로운 데이터도 동일한 scaler로 변환
new_samples_scaled = scaler.transform(new_samples)

# 예측
new_predictions = svm_model.predict(new_samples_scaled)
new_predictions_labels = label_encoder.inverse_transform(new_predictions)

# 예측 확률 (decision_function 사용)
decision_scores = svm_model.decision_function(new_samples_scaled)

# 결과 출력
print("\nNew Sample Predictions:")
for i in range(len(new_samples)):
    print(f"\nSample {i+1}:")
    print(f"  Predicted Class: {new_predictions_labels[i]}")
    print(f"  Decision Scores: {decision_scores[i]}")
    print(f"  Most confident class: {label_encoder.classes_[np.argmax(decision_scores[i])]}")

# 새로운 샘플 예측 결과를 DataFrame으로 정리
results_df = pd.DataFrame({
    'Sample_ID': [f'NewSample_{i+1}' for i in range(len(new_samples))],
    'Predicted_Class': new_predictions_labels,
    'Confidence_Score': [np.max(scores) for scores in decision_scores]
})

print("\n=== Summary of New Predictions ===")
print(results_df)
