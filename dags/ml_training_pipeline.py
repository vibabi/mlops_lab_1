import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator

ML_PROJECT_PATH = os.getenv("ML_PROJECT_PATH", "/opt/ml_project")
VENV_PYTHON = os.path.join(ML_PROJECT_PATH, "venv", "bin", "python")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ML_PROJECT_PATH}/mlflow.db")
F1_THRESHOLD = float(os.getenv("F1_THRESHOLD", "0.70"))

DEFAULT_ARGS = {
    "owner": "viktoriia",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_training_pipeline",
    description="Повний ML-пайплайн: підготовка даних → тренування → оцінка → реєстрація",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule_interval="@weekly",  
    catchup=False,
    tags=["mlops", "credit-card-fraud", "lab5"],
) as dag:

    check_data = BashOperator(
        task_id="check_data",
        bash_command=f"""
            echo "=== Перевірка доступності даних ==="
            if [ ! -f "{ML_PROJECT_PATH}/data/processed/train.csv" ]; then
                echo "ERROR: train.csv not found!"
                exit 1
            fi
            if [ ! -f "{ML_PROJECT_PATH}/data/processed/val.csv" ]; then
                echo "ERROR: val.csv not found!"
                exit 1
            fi
            if [ ! -f "{ML_PROJECT_PATH}/data/processed/test.csv" ]; then
                echo "ERROR: test.csv not found!"
                exit 1
            fi
            ROWS=$(wc -l < "{ML_PROJECT_PATH}/data/processed/train.csv")
            echo "train.csv: $ROWS рядків"
            echo "Дані доступні!"
        """,
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"""
            echo "=== Підготовка даних ==="
            cd {ML_PROJECT_PATH}
            # Перевіряємо чи є dvc.yaml зі стадією prepare
            if grep -q "prepare:" dvc.yaml 2>/dev/null; then
                echo "Запуск dvc repro prepare..."
                {VENV_PYTHON} -m dvc repro prepare --no-commit || true
            else
                echo "DVC стадія prepare не знайдена, дані вже підготовлені"
            fi
            echo "Підготовка даних завершена!"
        """,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"""
            echo "=== Тренування моделі ==="
            cd {ML_PROJECT_PATH}
            MLFLOW_TRACKING_URI="{MLFLOW_URI}" \\
            {VENV_PYTHON} src/train.py \\
                --n_estimators 167 \\
                --max_depth 13
            echo "Тренування завершено!"
            echo "Метрики:"
            cat metrics.json
        """,
    )

    def evaluate_and_branch(**kwargs):
        """
        Читає metrics.json та вирішує куди йти далі.
        Якщо f1 >= F1_THRESHOLD → register_model
        Інакше → stop_pipeline
        """
        metrics_path = os.path.join(ML_PROJECT_PATH, "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"ERROR: metrics.json not found at {metrics_path}")
            return "stop_pipeline"

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        f1 = float(metrics.get("f1", 0.0))
        accuracy = float(metrics.get("accuracy", 0.0))

        print(f"=== Оцінка моделі ===")
        print(f"F1-score:  {f1:.4f} (поріг: {F1_THRESHOLD})")
        print(f"Accuracy:  {accuracy:.4f}")

        kwargs["ti"].xcom_push(key="f1", value=f1)
        kwargs["ti"].xcom_push(key="accuracy", value=accuracy)
        kwargs["ti"].xcom_push(key="metrics", value=metrics)

        if f1 >= F1_THRESHOLD:
            print(f"Модель ПРОЙШЛА Quality Gate! f1={f1:.4f} >= {F1_THRESHOLD}")
            return "register_model"
        else:
            print(f"Модель НЕ пройшла Quality Gate: f1={f1:.4f} < {F1_THRESHOLD}")
            return "stop_pipeline"

    evaluate_model = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_and_branch,
        provide_context=True,
    )

    def register_model_fn(**kwargs):
        """Реєстрація найкращої моделі у MLflow Model Registry зі стадією Staging."""
        import mlflow
        import mlflow.sklearn
        import joblib

        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="evaluate_model", key="metrics")
        f1 = ti.xcom_pull(task_ids="evaluate_model", key="f1")

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("Credit_Card_Fraud_Detection_Pipeline")

        model_path = os.path.join(ML_PROJECT_PATH, "models", "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.pkl not found: {model_path}")

        model = joblib.load(model_path)

        with mlflow.start_run(run_name="dag_registered_model") as run:
            mlflow.log_metrics(metrics or {})
            mlflow.log_param("registered_by", "airflow_dag")
            mlflow.log_param("dag_id", "ml_training_pipeline")
            mlflow.sklearn.log_model(model, artifact_path="model")

            model_uri = f"runs:/{run.info.run_id}/model"

        try:
            mv = mlflow.register_model(model_uri, "CreditCardFraudDetector")
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name="CreditCardFraudDetector",
                version=mv.version,
                stage="Staging",
            )
            print(f"Модель зареєстрована: версія {mv.version} → Staging")
            print(f"F1-score: {f1:.4f}")
        except Exception as e:
            print(f"Реєстрація в Registry недоступна (file store): {e}")
            print("Модель залогована в MLflow як артефакт")

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_model_fn,
        provide_context=True,
    )

    stop_pipeline = EmptyOperator(
        task_id="stop_pipeline",
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    check_data >> prepare_data >> train_model >> evaluate_model
    evaluate_model >> [register_model, stop_pipeline]
    register_model >> end
    stop_pipeline >> end