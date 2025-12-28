"""
File untuk kriteria BASIC:
- Menggunakan MLflow AUTOLOG
- Disimpan LOKAL (tanpa DagsHub)

Perbaikan:
- Tambah metrik klasifikasi lengkap: accuracy, precision, recall, f1, roc_auc
- Simpan artefak: classification_report.txt + confusion_matrix.png (+ roc_curve.png opsional)
"""

import os
import warnings
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- 1. PREPARATION ---
print("\n" + "=" * 80)
print("MEMBANGUN MODEL MACHINE LEARNING - KRITERIA BASIC")
print("=" * 80)

# Load Dataset
print("\n--- Loading Data ---")
url = "https://raw.githubusercontent.com/mraihanbaa-cpu/Membangun-Model/main/top_1000_books_preprocessing.csv"
df = pd.read_csv(url)

# Pisahkan fitur dan target
X = df.drop(["bestseller_status"], axis=1)
y = df["bestseller_status"].astype(int)

print(f"✓ Dataset loaded: {df.shape}")
print(f"✓ Features: {X.shape[1]} columns")
print("✓ Target distribution:")
print(y.value_counts())

# Split data (stratify penting untuk klasifikasi tidak imbalance parah)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Training set: {X_train.shape}")
print(f"✓ Testing set: {X_test.shape}")

# --- 2. MLFLOW EXPERIMENT DENGAN AUTOLOG ---

# Set MLflow tracking URI ke lokal
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Book_Popularity_Basic")

# Folder artefak lokal (untuk file report & gambar)
ART_DIR = "artifacts_basic"
os.makedirs(ART_DIR, exist_ok=True)

print("\n--- Starting Training with MLflow Autolog ---")

# Aktifkan autolog SEBELUM start_run
mlflow.sklearn.autolog(log_models=True)

with mlflow.start_run(run_name="RandomForest_Autolog_Basic"):

    print("Training model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # --- 3. EVALUATION: METRICS ---
    y_pred = model.predict(X_test)

    # Untuk ROC-AUC perlu probabilitas (atau decision score)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

    # ROC-AUC (kalau proba tersedia)
    roc_auc = None
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)

    # Log metrics secara eksplisit (biar pasti masuk ke MLflow)
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_precision", prec)
    mlflow.log_metric("test_recall", rec)
    mlflow.log_metric("test_f1", f1)
    if roc_auc is not None:
        mlflow.log_metric("test_roc_auc", roc_auc)

    print("\n✓ Training Completed!")
    print(f"✓ Accuracy : {acc:.4f}")
    print(f"✓ Precision: {prec:.4f}")
    print(f"✓ Recall   : {rec:.4f}")
    print(f"✓ F1-score : {f1:.4f}")
    if roc_auc is not None:
        print(f"✓ ROC-AUC  : {roc_auc:.4f}")

    # --- 4. ARTIFACTS: CLASSIFICATION REPORT ---
    report_txt = classification_report(y_test, y_pred, digits=4)
    report_path = os.path.join(ART_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    mlflow.log_artifact(report_path, artifact_path="evaluation")

    # --- 5. ARTIFACTS: CONFUSION MATRIX (PNG) ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, values_format="d")
    ax.set_title("Confusion Matrix (Test Set)")
    cm_path = os.path.join(ART_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    mlflow.log_artifact(cm_path, artifact_path="evaluation")

    # --- 6. OPTIONAL ARTIFACT: ROC CURVE (PNG) ---
    if y_proba is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.set_title("ROC Curve (Test Set)")
        roc_path = os.path.join(ART_DIR, "roc_curve.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=200)
        plt.close(fig)

        mlflow.log_artifact(roc_path, artifact_path="evaluation")

    print(f"\n✓ Run ID: {mlflow.active_run().info.run_id}")

print("\n" + "=" * 80)
print("PELATIHAN SELESAI!")
print("=" * 80)
print("\nUntuk melihat hasil:")
print("1. Jalankan: mlflow ui")
print("2. Buka browser: http://127.0.0.1:5000")
print("3. Buka experiment 'Book_Popularity_Basic' -> pilih run -> lihat:")
print("   - Metrics: test_accuracy, test_precision, test_recall, test_f1, test_roc_auc")
print("   - Artifacts/evaluation/: classification_report.txt, confusion_matrix.png, roc_curve.png")
print("=" * 80)