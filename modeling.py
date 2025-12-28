"""
FINAL - MLflow to DagsHub (BASIC++)
- MLflow autolog
- Tracking ke DagsHub MLflow server
- Log metrics klasifikasi lengkap + artifacts evaluasi

WAJIB sebelum run (Windows PowerShell):
$env:MLFLOW_TRACKING_USERNAME="mraihanbaa-cpu"
$env:MLFLOW_TRACKING_PASSWORD="TOKEN_DAGSHUB_KAMU"
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


# =========================
# CONFIG
# =========================
DAGSHUB_USER = "mraihanbaa-cpu"
DAGSHUB_REPO = "Membangun-Model"
DAGSHUB_MLFLOW_URI = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"

EXPERIMENT_NAME = "Book_Popularity_Basic"
RUN_NAME = "RandomForest_Autolog_DagsHub"

DATA_URL = "https://raw.githubusercontent.com/mraihanbaa-cpu/Membangun-Model/main/top_1000_books_preprocessing.csv"

ART_DIR = "artifacts_basic"  # lokal sementara untuk bikin file png/txt


def ensure_credentials_or_fail() -> None:
    """
    DagsHub MLflow pakai basic auth.
    Kita pakai env var agar aman (tidak hardcode token di script).
    """
    u = os.getenv("MLFLOW_TRACKING_USERNAME")
    p = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not u or not p:
        raise RuntimeError(
            "Credential belum diset.\n"
            "Set dulu env var:\n"
            "  Windows PowerShell:\n"
            '    $env:MLFLOW_TRACKING_USERNAME="mraihanbaa-cpu"\n'
            '    $env:MLFLOW_TRACKING_PASSWORD="TOKEN_DAGSHUB_KAMU"\n'
            "  Linux/macOS:\n"
            '    export MLFLOW_TRACKING_USERNAME="mraihanbaa-cpu"\n'
            '    export MLFLOW_TRACKING_PASSWORD="TOKEN_DAGSHUB_KAMU"\n'
        )


def main():
    print("\n" + "=" * 80)
    print("MEMBANGUN MODEL MACHINE LEARNING - FINAL (MLflow -> DagsHub)")
    print("=" * 80)

    # 1) Credentials check
    ensure_credentials_or_fail()

    # 2) Load data
    print("\n--- Loading Data ---")
    df = pd.read_csv(DATA_URL)

    X = df.drop(["bestseller_status"], axis=1)
    y = df["bestseller_status"].astype(int)

    print(f"✓ Dataset loaded: {df.shape}")
    print(f"✓ Features: {X.shape[1]} columns")
    print("✓ Target distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✓ Training set: {X_train.shape}")
    print(f"✓ Testing set: {X_test.shape}")

    # 3) MLflow setup to DagsHub
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    os.makedirs(ART_DIR, exist_ok=True)

    print("\n--- Starting Training with MLflow Autolog (DagsHub) ---")

    # Autolog ON
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name=RUN_NAME) as run:

        # Optional: tag info biar rapi di UI
        mlflow.set_tag("platform", "dagshub")
        mlflow.set_tag("project", "Book_Popularity")
        mlflow.set_tag("task", "binary_classification")

        print("Training model...")

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # 4) Evaluate
        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

        roc_auc = None
        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba)

        # Log metrics (biar pasti muncul di DagsHub)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1", f1)
        if roc_auc is not None:
            mlflow.log_metric("test_roc_auc", roc_auc)

        print("\n✓ Training Completed!")
        print(f"✓ test_accuracy : {acc:.4f}")
        print(f"✓ test_precision: {prec:.4f}")
        print(f"✓ test_recall   : {rec:.4f}")
        print(f"✓ test_f1       : {f1:.4f}")
        if roc_auc is not None:
            print(f"✓ test_roc_auc  : {roc_auc:.4f}")

        # 5) Artifacts - classification report
        report_txt = classification_report(y_test, y_pred, digits=4)
        report_path = os.path.join(ART_DIR, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_txt)
        mlflow.log_artifact(report_path, artifact_path="evaluation")

        # 6) Artifacts - confusion matrix
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

        # 7) Artifacts - ROC curve (optional)
        if y_proba is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
            ax.set_title("ROC Curve (Test Set)")
            roc_path = os.path.join(ART_DIR, "roc_curve.png")
            plt.tight_layout()
            plt.savefig(roc_path, dpi=200)
            plt.close(fig)
            mlflow.log_artifact(roc_path, artifact_path="evaluation")

        print(f"\n✓ Run ID: {run.info.run_id}")
        print(f"✓ Tracking URI: {DAGSHUB_MLFLOW_URI}")

    print("\n" + "=" * 80)
    print("SELESAI! Cek di DagsHub:")
    print(DAGSHUB_MLFLOW_URI)
    print("=" * 80)


if __name__ == "__main__":
    main()