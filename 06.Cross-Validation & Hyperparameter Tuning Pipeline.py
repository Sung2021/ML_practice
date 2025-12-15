# %% [markdown]
# # 06. Cross-Validation & Hyperparameter Tuning Pipeline
# 
# **Learning Objectives:**
# - Implement various cross-validation strategies
# - Compare hyperparameter tuning methods (Grid, Random, Bayesian)
# - Analyze learning curves and model performance
# - Build reusable tuning pipeline
# 
# **Datasets:**
# - Classification: Cancer subtype prediction (gene expression, 3 subtypes, imbalanced)
# - Regression: Drug response prediction (proteomics, IC50 values, with outliers)

# %% [markdown]
# ## 1. Setup and Data Generation

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score,
    cross_validate,
    KFold,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# %%
# Generate realistic classification dataset (Cancer subtype prediction)
# Simulating gene expression data with 3 cancer subtypes
X_class, y_class = make_classification(
    n_samples=500,  # Typical clinical cohort size
    n_features=50,  # Gene expression features
    n_informative=15,  # Only subset are biomarkers
    n_redundant=10,  # Correlated genes
    n_repeated=5,  # Technical replicates
    n_classes=3,  # Cancer subtypes (e.g., Luminal A, Luminal B, Basal-like)
    n_clusters_per_class=1,
    weights=[0.5, 0.3, 0.2],  # Imbalanced classes (realistic)
    flip_y=0.05,  # 5% label noise (misdiagnosis)
    class_sep=0.8,  # Not perfectly separable
    random_state=42
)

# Add realistic feature scaling (log-transformed expression levels)
X_class = np.abs(X_class) + 0.1  # Ensure positive
X_class = np.log2(X_class + 1)  # Log transformation

# Generate realistic regression dataset (Drug response prediction)
# Simulating proteomics data predicting IC50 values
X_reg, y_reg = make_regression(
    n_samples=400,  # Drug screening dataset size
    n_features=50,  # Protein expression features
    n_informative=12,  # Key pathway proteins
    noise=15,  # Measurement noise
    random_state=42
)

# Transform to realistic IC50 range (0.1 to 100 μM, log-scale)
y_reg = (y_reg - y_reg.min()) / (y_reg.max() - y_reg.min())  # Normalize to 0-1
y_reg = 0.1 * np.exp(y_reg * np.log(1000))  # Transform to 0.1-100 range

# Add outliers (5% of samples, common in biological data)
n_outliers = int(0.05 * len(y_reg))
outlier_idx = np.random.choice(len(y_reg), n_outliers, replace=False)
y_reg[outlier_idx] *= np.random.uniform(2, 5, n_outliers)  # 2-5x higher values

# Add realistic feature scaling (z-score normalized proteomics)
from sklearn.preprocessing import StandardScaler
scaler_reg = StandardScaler()
X_reg = scaler_reg.fit_transform(X_reg)

print("=== Classification Dataset (Cancer Subtype) ===")
print(f"Shape: {X_class.shape}")
print(f"Class distribution: {dict(zip(*np.unique(y_class, return_counts=True)))}")
print(f"Class proportions: {np.bincount(y_class) / len(y_class)}")
print(f"Feature range: [{X_class.min():.2f}, {X_class.max():.2f}]")

print("\n=== Regression Dataset (Drug Response IC50) ===")
print(f"Shape: {X_reg.shape}")
print(f"IC50 range: [{y_reg.min():.2f}, {y_reg.max():.2f}] μM")
print(f"IC50 median: {np.median(y_reg):.2f} μM")
print(f"IC50 std: {np.std(y_reg):.2f}")
print(f"Number of outliers: {n_outliers}")

# %%
# Visualize realistic data characteristics
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Classification: Class distribution
axes[0, 0].bar(range(3), np.bincount(y_class), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0, 0].set_xlabel('Cancer Subtype')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Imbalanced Class Distribution')
axes[0, 0].set_xticks(range(3))
axes[0, 0].set_xticklabels(['Type 1 (50%)', 'Type 2 (30%)', 'Type 3 (20%)'])

# Classification: Feature distribution (first 2 features)
for class_id in range(3):
    mask = y_class == class_id
    axes[0, 1].scatter(X_class[mask, 0], X_class[mask, 1], 
                       alpha=0.6, label=f'Type {class_id+1}', s=30)
axes[0, 1].set_xlabel('Gene 1 (log2 expression)')
axes[0, 1].set_ylabel('Gene 2 (log2 expression)')
axes[0, 1].set_title('Feature Space (Not Perfectly Separable)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Classification: Feature correlation heatmap
corr_class = np.corrcoef(X_class.T)[:10, :10]  # First 10 features
im1 = axes[0, 2].imshow(corr_class, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
axes[0, 2].set_title('Feature Correlation\n(Redundant genes)')
axes[0, 2].set_xlabel('Feature')
axes[0, 2].set_ylabel('Feature')
plt.colorbar(im1, ax=axes[0, 2])

# Regression: IC50 distribution with outliers
axes[1, 0].hist(y_reg, bins=30, color='skyblue', edgecolor='black')
axes[1, 0].axvline(np.median(y_reg), color='red', linestyle='--', label='Median')
axes[1, 0].set_xlabel('IC50 (μM)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('IC50 Distribution (with outliers)')
axes[1, 0].legend()
axes[1, 0].set_yscale('log')

# Regression: Log-scale IC50 (more normal)
axes[1, 1].hist(np.log10(y_reg), bins=30, color='lightcoral', edgecolor='black')
axes[1, 1].set_xlabel('log10(IC50)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Log-transformed IC50\n(Closer to normal)')

# Regression: Feature vs Response
axes[1, 2].scatter(X_reg[:, 0], y_reg, alpha=0.5, s=30)
axes[1, 2].set_xlabel('Protein 1 (z-score)')
axes[1, 2].set_ylabel('IC50 (μM)')
axes[1, 2].set_title('Feature-Response Relationship')
axes[1, 2].set_yscale('log')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Cross-Validation Strategies

# %%
# 2.1 K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf_model, X_class, y_class, cv=kfold, scoring='accuracy')

print("K-Fold Cross-Validation Results:")
print(f"Scores per fold: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# %%
# 2.2 Stratified K-Fold (maintains class distribution)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_strat = cross_val_score(rf_model, X_class, y_class, cv=skfold, scoring='accuracy')

print("\nStratified K-Fold Cross-Validation Results:")
print(f"Scores per fold: {cv_scores_strat}")
print(f"Mean accuracy: {cv_scores_strat.mean():.4f} (+/- {cv_scores_strat.std():.4f})")

# %%
# 2.3 Cross-validate with multiple metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

cv_results = cross_validate(
    rf_model, 
    X_class, 
    y_class, 
    cv=skfold, 
    scoring=scoring,
    return_train_score=True
)

print("\nMultiple Metrics Cross-Validation:")
for metric in scoring:
    test_key = f'test_{metric}'
    train_key = f'train_{metric}'
    print(f"{metric}:")
    print(f"  Train: {cv_results[train_key].mean():.4f} (+/- {cv_results[train_key].std():.4f})")
    print(f"  Test:  {cv_results[test_key].mean():.4f} (+/- {cv_results[test_key].std():.4f})")

# %% [markdown]
# ## 3. Hyperparameter Tuning Methods

# %%
# Split data for tuning
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# %% [markdown]
# ### 3.1 Grid Search CV

# %%
# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")

# %%
# Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print("\nGrid Search Results:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")

# %%
# Analyze grid search results
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values('rank_test_score')

print("\nTop 5 configurations:")
print(results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].head())

# %% [markdown]
# ### 3.2 Randomized Search CV

# %%
# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

# %%
# Randomized Search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True
)

random_search.fit(X_train, y_train)

print("\nRandomized Search Results:")
print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
print(f"Test score: {random_search.score(X_test, y_test):.4f}")

# %%
# Compare Grid vs Random Search
print("\nComparison:")
print(f"Grid Search    - Best CV: {grid_search.best_score_:.4f}, Test: {grid_search.score(X_test, y_test):.4f}")
print(f"Random Search  - Best CV: {random_search.best_score_:.4f}, Test: {random_search.score(X_test, y_test):.4f}")

# %% [markdown]
# ### 3.3 Bayesian Optimization (using Optuna)

# %%
# Install optuna if needed: pip install optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0)
        }
        
        model = RandomForestClassifier(**params, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        return scores.mean()
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    # Optimize
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print("\nBayesian Optimization (Optuna) Results:")
    print(f"Best parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    # Train final model
    optuna_model = RandomForestClassifier(**study.best_params, random_state=42)
    optuna_model.fit(X_train, y_train)
    print(f"Test score: {optuna_model.score(X_test, y_test):.4f}")
    
except ImportError:
    print("Optuna not installed. Skip Bayesian optimization.")
    print("Install with: pip install optuna")

# %% [markdown]
# ## 4. Learning Curves Analysis

# %%
# 4.1 Learning curve for sample size
train_sizes = np.linspace(0.1, 1.0, 10)

train_sizes_abs, train_scores, val_scores = learning_curve(
    RandomForestClassifier(**grid_search.best_params_, random_state=42),
    X_train,
    y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# %%
# Plot learning curve
plt.figure(figsize=(10, 6))

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.plot(train_sizes_abs, train_mean, label='Training score', marker='o')
plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)

plt.plot(train_sizes_abs, val_mean, label='Validation score', marker='s')
plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2)

plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve - Training vs Validation Score')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Diagnosis
gap = train_mean[-1] - val_mean[-1]
print(f"\nLearning Curve Diagnosis:")
print(f"Final training accuracy: {train_mean[-1]:.4f}")
print(f"Final validation accuracy: {val_mean[-1]:.4f}")
print(f"Gap: {gap:.4f}")

if gap > 0.1:
    print("→ High variance (overfitting) - consider regularization or more data")
elif val_mean[-1] < 0.7:
    print("→ High bias (underfitting) - consider more complex model or features")
else:
    print("→ Good fit - model generalizes well")

# %% [markdown]
# ## 5. Validation Curve (Parameter Impact)

# %%
from sklearn.model_selection import validation_curve

# Analyze max_depth impact
param_range = [5, 10, 15, 20, 25, 30]

train_scores_vc, val_scores_vc = validation_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train,
    y_train,
    param_name='max_depth',
    param_range=param_range,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# %%
# Plot validation curve
plt.figure(figsize=(10, 6))

train_mean_vc = train_scores_vc.mean(axis=1)
train_std_vc = train_scores_vc.std(axis=1)
val_mean_vc = val_scores_vc.mean(axis=1)
val_std_vc = val_scores_vc.std(axis=1)

plt.plot(param_range, train_mean_vc, label='Training score', marker='o')
plt.fill_between(param_range, train_mean_vc - train_std_vc, train_mean_vc + train_std_vc, alpha=0.2)

plt.plot(param_range, val_mean_vc, label='Validation score', marker='s')
plt.fill_between(param_range, val_mean_vc - val_std_vc, val_mean_vc + val_std_vc, alpha=0.2)

plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Validation Curve - max_depth Impact')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Find optimal parameter
optimal_idx = val_mean_vc.argmax()
print(f"\nOptimal max_depth: {param_range[optimal_idx]}")
print(f"Validation accuracy: {val_mean_vc[optimal_idx]:.4f}")

# %% [markdown]
# ## 6. Pipeline Integration (Preventing Data Leakage)

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Create preprocessing + model pipeline
pipeline_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter grid with pipeline steps
param_grid_pipeline = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, 15],
    'classifier__min_samples_split': [2, 5, 10]
}

# %%
# Grid search with pipeline
grid_pipeline = GridSearchCV(
    pipeline_clf,
    param_grid_pipeline,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_pipeline.fit(X_train, y_train)

print("Pipeline Grid Search Results:")
print(f"Best parameters: {grid_pipeline.best_params_}")
print(f"Best CV score: {grid_pipeline.best_score_:.4f}")
print(f"Test score: {grid_pipeline.score(X_test, y_test):.4f}")

# %%
# Why pipeline matters: Data leakage example
print("\n=== Data Leakage Demonstration ===")

# WRONG: Scaling before split (data leakage)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled_wrong = scaler.fit_transform(X_class)  # Uses info from entire dataset
X_train_wrong, X_test_wrong, y_train_wrong, y_test_wrong = train_test_split(
    X_scaled_wrong, y_class, test_size=0.2, random_state=42
)

model_wrong = RandomForestClassifier(n_estimators=100, random_state=42)
model_wrong.fit(X_train_wrong, y_train_wrong)
score_wrong = model_wrong.score(X_test_wrong, y_test_wrong)

# CORRECT: Pipeline scales within CV folds
pipeline_correct = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train_correct, X_test_correct, y_train_correct, y_test_correct = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

pipeline_correct.fit(X_train_correct, y_train_correct)
score_correct = pipeline_correct.score(X_test_correct, y_test_correct)

print(f"Wrong approach (leakage): {score_wrong:.4f}")
print(f"Correct approach (pipeline): {score_correct:.4f}")
print(f"Difference: {abs(score_wrong - score_correct):.4f}")
print("\nNote: Leakage often inflates performance estimate")

# %% [markdown]
# ## 7. Regression Problem Tuning

# %%
# Split regression data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# %%
# Regression pipeline
pipeline_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Parameter grid for regression
param_grid_reg = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__max_depth': [3, 5, 7],
    'regressor__min_samples_split': [2, 5, 10]
}

print(f"Total combinations: {np.prod([len(v) for v in param_grid_reg.values()])}")

# %%
# Grid search for regression
grid_reg = GridSearchCV(
    pipeline_reg,
    param_grid_reg,
    cv=5,
    scoring='neg_mean_squared_error',  # Regression metric
    n_jobs=-1,
    verbose=1
)

grid_reg.fit(X_train_reg, y_train_reg)

# Get predictions
y_pred_reg = grid_reg.predict(X_test_reg)
test_mse = mean_squared_error(y_test_reg, y_pred_reg)
test_rmse = np.sqrt(test_mse)

print("\nRegression Tuning Results:")
print(f"Best parameters: {grid_reg.best_params_}")
print(f"Best CV MSE: {-grid_reg.best_score_:.2f}")
print(f"Test MSE: {test_mse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# %%
# Compare regression tuning methods
param_dist_reg = {
    'regressor__n_estimators': randint(50, 300),
    'regressor__learning_rate': uniform(0.01, 0.3),
    'regressor__max_depth': randint(3, 10),
    'regressor__min_samples_split': randint(2, 20)
}

random_reg = RandomizedSearchCV(
    pipeline_reg,
    param_dist_reg,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_reg.fit(X_train_reg, y_train_reg)
y_pred_random = random_reg.predict(X_test_reg)
test_mse_random = mean_squared_error(y_test_reg, y_pred_random)

print("\nRegression Tuning Comparison:")
print(f"Grid Search   - CV MSE: {-grid_reg.best_score_:.2f}, Test MSE: {test_mse:.2f}")
print(f"Random Search - CV MSE: {-random_reg.best_score_:.2f}, Test MSE: {test_mse_random:.2f}")

# %%
# Visualize regression predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], 
         [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f'Grid Search (RMSE: {test_rmse:.2f})')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test_reg - y_pred_reg
plt.scatter(y_pred_reg, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Nested Cross-Validation (Unbiased Performance Estimation)

# %%
# Nested CV: outer loop for performance estimation, inner loop for tuning
from sklearn.model_selection import cross_val_score

# Simplified parameter grid for demonstration
param_grid_nested = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, 15]
}

# Inner CV: hyperparameter tuning (3-fold)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
clf_nested = GridSearchCV(
    pipeline_clf, 
    param_grid_nested, 
    cv=inner_cv,
    scoring='accuracy',
    n_jobs=-1
)

# Outer CV: performance estimation (5-fold)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nested_scores = cross_val_score(
    clf_nested, 
    X_train, 
    y_train, 
    cv=outer_cv,
    scoring='accuracy',
    n_jobs=-1
)

print("=== Nested Cross-Validation ===")
print(f"Nested CV scores: {nested_scores}")
print(f"Mean score: {nested_scores.mean():.4f} (+/- {nested_scores.std():.4f})")

# %%
# Compare with non-nested CV (biased estimate)
non_nested_clf = GridSearchCV(
    pipeline_clf,
    param_grid_nested,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
non_nested_clf.fit(X_train, y_train)

print("\n=== Non-Nested vs Nested CV Comparison ===")
print(f"Non-nested CV score: {non_nested_clf.best_score_:.4f} (optimistically biased)")
print(f"Nested CV score:     {nested_scores.mean():.4f} (unbiased estimate)")
print(f"Bias:                {non_nested_clf.best_score_ - nested_scores.mean():.4f}")
print("\nNested CV provides more realistic performance expectation on unseen data")

# %%
# Visualize nested CV structure
fig, ax = plt.subplots(figsize=(12, 6))

outer_folds = 5
inner_folds = 3

y_pos = 0
for outer in range(outer_folds):
    # Outer fold
    ax.barh(y_pos, 1, height=0.8, color='lightblue', edgecolor='black', linewidth=2)
    ax.text(0.5, y_pos, f'Outer Fold {outer+1}', ha='center', va='center', fontsize=10, fontweight='bold')
    y_pos += 1
    
    # Inner folds
    for inner in range(inner_folds):
        ax.barh(y_pos, 0.9, left=0.05, height=0.6, color='lightcoral', edgecolor='gray')
        ax.text(0.5, y_pos, f'Inner {inner+1}', ha='center', va='center', fontsize=8)
        y_pos += 0.7
    
    y_pos += 0.5

ax.set_ylim(0, y_pos)
ax.set_xlim(0, 1)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title('Nested Cross-Validation Structure\n(Outer: Performance Estimation, Inner: Hyperparameter Tuning)', 
             fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Reusable Tuning Pipeline

# %%
class HyperparameterTuner:
    """Reusable hyperparameter tuning class"""
    
    def __init__(self, model, param_grid, cv=5, scoring='accuracy'):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_model = None
        self.results = {}
    
    def grid_search(self, X, y):
        """Grid search tuning"""
        gs = GridSearchCV(
            self.model, self.param_grid, cv=self.cv, 
            scoring=self.scoring, n_jobs=-1, return_train_score=True
        )
        gs.fit(X, y)
        self.best_model = gs.best_estimator_
        self.results['grid'] = {
            'best_params': gs.best_params_,
            'best_score': gs.best_score_,
            'cv_results': pd.DataFrame(gs.cv_results_)
        }
        return gs.best_params_, gs.best_score_
    
    def random_search(self, X, y, n_iter=50):
        """Random search tuning"""
        rs = RandomizedSearchCV(
            self.model, self.param_grid, n_iter=n_iter,
            cv=self.cv, scoring=self.scoring, n_jobs=-1,
            random_state=42, return_train_score=True
        )
        rs.fit(X, y)
        self.best_model = rs.best_estimator_
        self.results['random'] = {
            'best_params': rs.best_params_,
            'best_score': rs.best_score_,
            'cv_results': pd.DataFrame(rs.cv_results_)
        }
        return rs.best_params_, rs.best_score_
    
    def evaluate(self, X_test, y_test):
        """Evaluate best model"""
        if self.best_model is None:
            raise ValueError("Run tuning first")
        score = self.best_model.score(X_test, y_test)
        return score
    
    def plot_results(self, method='grid', top_n=10):
        """Plot top configurations"""
        if method not in self.results:
            raise ValueError(f"No results for {method}")
        
        df = self.results[method]['cv_results']
        df_sorted = df.sort_values('rank_test_score').head(top_n)
        
        plt.figure(figsize=(12, 6))
        x = range(len(df_sorted))
        plt.bar(x, df_sorted['mean_test_score'], yerr=df_sorted['std_test_score'], alpha=0.7)
        plt.xlabel('Configuration Rank')
        plt.ylabel('Mean CV Score')
        plt.title(f'Top {top_n} Configurations ({method.capitalize()} Search)')
        plt.xticks(x, range(1, top_n+1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# %%
# Use tuning pipeline with sklearn Pipeline
tuner = HyperparameterTuner(
    model=pipeline_clf,
    param_grid=param_grid_pipeline,
    cv=5,
    scoring='accuracy'
)

# Grid search
best_params_g, best_score_g = tuner.grid_search(X_train, y_train)
test_score_g = tuner.evaluate(X_test, y_test)

print("Pipeline Grid Search Results:")
print(f"Best params: {best_params_g}")
print(f"Best CV score: {best_score_g:.4f}")
print(f"Test score: {test_score_g:.4f}")

# %%
# Plot results
tuner.plot_results(method='grid', top_n=10)

# %%
# Regression tuning with custom class
tuner_reg = HyperparameterTuner(
    model=pipeline_reg,
    param_grid=param_grid_reg,
    cv=5,
    scoring='neg_mean_squared_error'
)

best_params_reg, best_score_reg = tuner_reg.grid_search(X_train_reg, y_train_reg)
test_score_reg = tuner_reg.evaluate(X_test_reg, y_test_reg)

print("\nRegression Pipeline Tuning Results:")
print(f"Best params: {best_params_reg}")
print(f"Best CV MSE: {-best_score_reg:.2f}")
print(f"Test MSE: {-test_score_reg:.2f}")

# %% [markdown]
# ## 10. Summary and Best Practices
# 
# **Cross-Validation:**
# - Use StratifiedKFold for classification (maintains class distribution)
# - 5-10 folds typical, balance between bias/variance and computation
# - Use cross_validate for multiple metrics simultaneously
# 
# **Hyperparameter Tuning:**
# - Grid Search: exhaustive but expensive, good for small search spaces
# - Random Search: efficient for large spaces, often 90% of Grid performance with 10% cost
# - Bayesian: most efficient, learns from previous trials
# 
# **Pipeline Integration:**
# - Always use sklearn.pipeline.Pipeline to prevent data leakage
# - Scaling/encoding happens within CV folds, not before
# - Use parameter names like 'classifier__n_estimators' for pipeline steps
# 
# **Learning Curves:**
# - Diagnose overfitting (high gap) vs underfitting (low performance)
# - Determine if more data would help
# - Validation curves show individual parameter impact
# 
# **Nested Cross-Validation:**
# - Outer loop: unbiased performance estimation
# - Inner loop: hyperparameter tuning
# - Provides realistic expectation on truly unseen data
# - Non-nested CV is optimistically biased
# 
# **Regression vs Classification:**
# - Use appropriate metrics: neg_mean_squared_error for regression, accuracy/f1 for classification
# - Residual plots help diagnose regression model fit
# - Both benefit equally from proper CV and pipeline usage
# 
# **Practical Tips:**
# - Start with Random Search to narrow range, then Grid Search
# - Monitor train/test gap for overfitting
# - Use nested CV for final unbiased performance report
# - Consider computational cost vs performance gain
# - Always prevent data leakage with proper pipeline usage

# %%
