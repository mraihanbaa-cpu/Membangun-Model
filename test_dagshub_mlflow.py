import os
import mlflow

print("=== DAGSHUB MLFLOW TEST ===")
print("USER:", os.getenv("MLFLOW_TRACKING_USERNAME"))
print("PASS_SET:", bool(os.getenv("MLFLOW_TRACKING_PASSWORD")))

mlflow.set_tracking_uri("https://dagshub.com/mraihanbaa-cpu/Membangun-Model.mlflow")
print("URI:", mlflow.get_tracking_uri())

mlflow.set_experiment("TEST_CONNECTION")

with mlflow.start_run(run_name="ping_test"):
    mlflow.log_metric("ok", 1)

print("DONE - run should appear in DagsHub")