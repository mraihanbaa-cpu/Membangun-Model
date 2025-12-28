"""
File untuk kriteria BASIC:
- Menggunakan MLflow AUTOLOG
- Disimpan LOKAL (tanpa DagsHub)
"""

# Install library yang dibutuhkan
# !pip install pandas numpy scikit-learn mlflow matplotlib seaborn

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Training Completed!")
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Run ID: {mlflow.active_run().info.run_id}")
    
    # Autolog akan mencatat:
    # - Parameters (n_estimators, random_state, dll)
    # - Metrics (training_score, dll)
    # - Model artifacts (folder model/, estimator.html)
    # - Feature importances
    
print("\n" + "="*80)
print("PELATIHAN SELESAI!")
print("="*80)
print("\nUntuk melihat hasil:")
print("1. Jalankan: mlflow ui")
print("2. Buka browser: http://127.0.0.1:5000")
print("3. Screenshot MLflow UI untuk artefak submission")
print("="*80)