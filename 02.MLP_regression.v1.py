import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================
# 임상 데이터 생성
# ============================================
np.random.seed(42)
torch.manual_seed(42)

n_samples = 500
n_features = 30  # 임상 지표 (나이, 혈압, 혈당, 바이오마커 등)

# 임상 특징 생성
X = np.random.randn(n_samples, n_features)

# 치료 반응 점수 생성 (0-100 사이, 일부 특징과 비선형 관계)
true_weights = np.random.randn(n_features) * 2
interaction_term = X[:, 0] * X[:, 5]  # 특징 간 상호작용
nonlinear_term = np.sin(X[:, 10]) * 3
y = (X @ true_weights + interaction_term + nonlinear_term + 
     np.random.randn(n_samples) * 5)

# 0-100 범위로 스케일링
y = (y - y.min()) / (y.max() - y.min()) * 100

# 데이터프레임 생성
feature_names = [f'Clinical_Feature_{i+1}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['Response_Score'] = y

print("Data Overview:")
print(df.head())
print(f"\nData shape: {df.shape}")
print(f"\nResponse Score statistics:")
print(df['Response_Score'].describe())

# ============================================
# Train-test split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# 스케일링
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# PyTorch 텐서로 변환
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

# DataLoader 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ============================================
# MLP 모델 정의
# ============================================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()

# 모델 초기화
input_dim = n_features
hidden_dims = [128, 64, 32]
model = MLPRegressor(input_dim, hidden_dims, dropout_rate=0.3).to(device)

print("\n=== Model Architecture ===")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================
# 학습 설정
# ============================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=10)

# ============================================
# 학습 함수
# ============================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, X, y, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # 원래 스케일로 복원
        outputs_numpy = scaler_y.inverse_transform(
            outputs.cpu().numpy().reshape(-1, 1)
        ).flatten()
        y_numpy = scaler_y.inverse_transform(
            y.cpu().numpy().reshape(-1, 1)
        ).flatten()
        
        mse = mean_squared_error(y_numpy, outputs_numpy)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_numpy, outputs_numpy)
        r2 = r2_score(y_numpy, outputs_numpy)
    
    return loss.item(), mse, rmse, mae, r2

# ============================================
# 모델 학습
# ============================================
print("\n=== Training Model ===")

n_epochs = 200
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 20

for epoch in range(n_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_mse, val_rmse, val_mae, val_r2 = evaluate(
        model, X_test_tensor, y_test_tensor, criterion
    )
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val R2: {val_r2:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# 최적 모델 로드
model.load_state_dict(torch.load('best_model.pth'))

# ============================================
# 학습 곡선 시각화
# ============================================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================
# 최종 성능 평가
# ============================================
print("\n=== Final Model Performance ===")

_, test_mse, test_rmse, test_mae, test_r2 = evaluate(
    model, X_test_tensor, y_test_tensor, criterion
)

print(f"Test MSE: {test_mse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R²: {test_r2:.4f}")

# 예측 vs 실제 시각화
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor)
    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.cpu().numpy().reshape(-1, 1)
    ).flatten()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2)
plt.xlabel('True Response Score')
plt.ylabel('Predicted Response Score')
plt.title(f'Predictions vs True Values (R²={test_r2:.3f})')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Response Score')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# Cross-validation으로 robustness 확인
# ============================================
print("\n=== Cross-Validation ===")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_scores = []
cv_rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
    print(f"\nFold {fold+1}/5")
    
    X_fold_train = torch.FloatTensor(X_train_scaled[train_idx]).to(device)
    y_fold_train = torch.FloatTensor(y_train_scaled[train_idx]).to(device)
    X_fold_val = torch.FloatTensor(X_train_scaled[val_idx]).to(device)
    y_fold_val = torch.FloatTensor(y_train_scaled[val_idx]).to(device)
    
    fold_dataset = TensorDataset(X_fold_train, y_fold_train)
    fold_loader = DataLoader(fold_dataset, batch_size=32, shuffle=True)
    
    fold_model = MLPRegressor(input_dim, hidden_dims, dropout_rate=0.3).to(device)
    fold_optimizer = optim.Adam(fold_model.parameters(), lr=0.001, weight_decay=1e-5)
    
    for epoch in range(100):
        train_epoch(fold_model, fold_loader, criterion, fold_optimizer)
    
    _, _, fold_rmse, _, fold_r2 = evaluate(
        fold_model, X_fold_val, y_fold_val, criterion
    )
    
    cv_r2_scores.append(fold_r2)
    cv_rmse_scores.append(fold_rmse)
    
    print(f"  Val R²: {fold_r2:.4f}, Val RMSE: {fold_rmse:.4f}")

print(f"\nCV R² scores: {cv_r2_scores}")
print(f"Mean CV R²: {np.mean(cv_r2_scores):.4f} (+/- {np.std(cv_r2_scores):.4f})")
print(f"Mean CV RMSE: {np.mean(cv_rmse_scores):.4f} (+/- {np.std(cv_rmse_scores):.4f})")

# ============================================
# 새로운 환자 데이터로 예측
# ============================================
print("\n=== Predicting on New Patients ===")

# 새로운 환자 5명 생성
new_patient_1 = np.random.randn(1, n_features)
new_patient_2 = np.random.randn(1, n_features) * 1.5
new_patient_3 = np.random.randn(1, n_features) * 0.5
new_patient_4 = np.random.randn(1, n_features) + 1
new_patient_5 = np.random.randn(1, n_features) - 1

new_patients = np.vstack([new_patient_1, new_patient_2, new_patient_3, 
                          new_patient_4, new_patient_5])

# 스케일링 및 예측
new_patients_scaled = scaler_X.transform(new_patients)
new_patients_tensor = torch.FloatTensor(new_patients_scaled).to(device)

model.eval()
with torch.no_grad():
    new_predictions_scaled = model(new_patients_tensor)
    new_predictions = scaler_y.inverse_transform(
        new_predictions_scaled.cpu().numpy().reshape(-1, 1)
    ).flatten()

# 결과 출력
print("\nNew Patient Predictions:")
for i in range(len(new_patients)):
    print(f"\nPatient {i+1}:")
    print(f"  Predicted Response Score: {new_predictions[i]:.2f}")

# 결과 정리
results_df = pd.DataFrame({
    'Patient_ID': [f'Patient_{i+1}' for i in range(len(new_patients))],
    'Predicted_Response_Score': np.round(new_predictions, 2)
})

print("\n=== Summary of New Predictions ===")
print(results_df)

print("\n=== Final Summary ===")
print(f"Model: MLP with {hidden_dims} hidden layers")
print(f"Test R²: {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"CV R² (mean): {np.mean(cv_r2_scores):.4f}")


