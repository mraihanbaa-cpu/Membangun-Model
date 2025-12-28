"""
File untuk kriteria BASIC:
- Menggunakan MLflow AUTOLOG
- Disimpan LOKAL (tanpa DagsHub)
- Tanpa hyperparameter tuning
- Ditambah 3 artefak manual: Feature Importance, Classification Report, Precision-Recall Curve
"""

# Install library yang dibutuhkan
# !pip install pandas numpy scikit-learn mlflow matplotlib seaborn

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, 
    precision_recall_curve, auc
)
import warnings

warnings.filterwarnings('ignore')

# --- 1. PREPARATION ---
print("\n" + "="*80)
print("MEMBANGUN MODEL MACHINE LEARNING - KRITERIA BASIC")
print("="*80)

# Load Dataset
print("\n--- Loading Data ---")
url = "https://raw.githubusercontent.com/mraihanbaa-cpu/Membangun-Model/main/top_1000_books_preprocessing.csv"
df = pd.read_csv(url)

# Pisahkan fitur dan target
X = df.drop(['bestseller_status'], axis=1)
y = df['bestseller_status'].astype(int)

print(f"✓ Dataset loaded: {df.shape}")
print(f"✓ Features: {X.shape[1]} columns")
print(f"✓ Target distribution:")
print(y.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Training set: {X_train.shape}")
print(f"✓ Testing set: {X_test.shape}")

# --- 2. MLFLOW EXPERIMENT DENGAN AUTOLOG ---

# Set MLflow tracking URI ke lokal (default)
mlflow.set_tracking_uri("file:./mlruns")

# Set experiment name
mlflow.set_experiment("Book_Popularity_Basic")

print("\n--- Starting Training with MLflow Autolog ---")

# PENTING: Aktifkan autolog SEBELUM start_run
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Autolog_Basic"):
    
    print("Training model...")
    
    # Inisialisasi dan latih model
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42
    )
    
    # Fit model - autolog akan mencatat semua informasi secara otomatis
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Training Completed!")
    print(f"✓ Accuracy: {accuracy:.4f}")
    
    # Autolog akan mencatat:
    # - Parameters (n_estimators, random_state, dll)
    # - Metrics (training_score, dll)
    # - Model artifacts (folder model/, estimator.html)
    # - Feature importances
    
    # --- 3. TAMBAHAN ARTEFAK MANUAL (OPSIONAL UNTUK BASIC) ---
    print("\n--- Creating Additional Artifacts ---")
    
    # ARTEFAK 1: Feature Importance Plot
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    top_n = min(15, len(feature_importance))
    sns.barplot(data=feature_importance.head(top_n), 
                x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importances - Random Forest')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=100, bbox_inches='tight')
    mlflow.log_artifact("feature_importance.png")
    plt.close()
    print("✓ Artifact 1: Feature Importance saved")
    
    # ARTEFAK 2: Classification Report (Text File)
    report = classification_report(y_test, y_pred, 
                                   target_names=['Non-Bestseller', 'Bestseller'],
                                   digits=4)
    
    with open("classification_report.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n\n" + "="*60 + "\n")
        f.write("MODEL INFORMATION\n")
        f.write("="*60 + "\n")
        f.write(f"Model: Random Forest Classifier\n")
        f.write(f"N Estimators: 100\n")
        f.write(f"Random State: 42\n")
        f.write(f"Test Size: 20%\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
    
    mlflow.log_artifact("classification_report.txt")
    print("✓ Artifact 2: Classification Report saved")
    
    # ARTEFAK 3: Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, 
             label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png", dpi=100, bbox_inches='tight')
    mlflow.log_artifact("precision_recall_curve.png")
    plt.close()
    print("✓ Artifact 3: Precision-Recall Curve saved")
    
    # Summary
    print(f"\n✓ Run ID: {mlflow.active_run().info.run_id}")
    
print("\n" + "="*80)
print("PELATIHAN SELESAI!")
print("="*80)
print("\nArtifacts yang berhasil dibuat:")
print("1. Model folder (dari autolog)")
print("2. estimator.html (dari autolog)")
print("3. feature_importance.png (manual)")
print("4. classification_report.txt (manual)")
print("5. precision_recall_curve.png (manual)")
print("\nUntuk melihat hasil:")
print("1. Jalankan: mlflow ui")
print("2. Buka browser: http://localhost:5000")
print("3. Screenshot MLflow UI untuk submission")
print("="*80)