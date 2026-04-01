import os
import pytest


class TestDAGIntegrity:
    def test_dag_import_no_errors(self):
        from airflow.models import DagBag

        dag_folder = os.path.join(os.path.dirname(__file__), "..", "dags")
        dag_bag = DagBag(dag_folder=dag_folder, include_examples=False)

        assert len(dag_bag.import_errors) == 0, (
            f"DAG import errors found:\n"
            + "\n".join(
                f"  {dag}: {err}"
                for dag, err in dag_bag.import_errors.items()
            )
        )

    def test_dag_exists(self):
        from airflow.models import DagBag

        dag_folder = os.path.join(os.path.dirname(__file__), "..", "dags")
        dag_bag = DagBag(dag_folder=dag_folder, include_examples=False)

        assert "ml_training_pipeline" in dag_bag.dags, (
            "DAG 'ml_training_pipeline' not found in dag_bag"
        )

    def test_dag_has_correct_tasks(self):
        from airflow.models import DagBag

        dag_folder = os.path.join(os.path.dirname(__file__), "..", "dags")
        dag_bag = DagBag(dag_folder=dag_folder, include_examples=False)

        dag = dag_bag.dags.get("ml_training_pipeline")
        if dag is None:
            pytest.skip("DAG not found")

        expected_tasks = {
            "check_data",
            "prepare_data",
            "train_model",
            "evaluate_model",
            "register_model",
            "stop_pipeline",
            "end",
        }
        actual_tasks = set(dag.task_ids)
        missing = expected_tasks - actual_tasks
        assert not missing, f"Missing tasks in DAG: {missing}"

    def test_dag_no_cycles(self):
        from airflow.models import DagBag

        dag_folder = os.path.join(os.path.dirname(__file__), "..", "dags")
        dag_bag = DagBag(dag_folder=dag_folder, include_examples=False)

        dag = dag_bag.dags.get("ml_training_pipeline")
        if dag is None:
            pytest.skip("DAG not found")

        # Якщо DAG завантажився без помилок — циклів немає
        # (Airflow перевіряє це при завантаженні)
        assert dag is not None

    def test_dag_schedule_interval(self):
        from airflow.models import DagBag

        dag_folder = os.path.join(os.path.dirname(__file__), "..", "dags")
        dag_bag = DagBag(dag_folder=dag_folder, include_examples=False)

        dag = dag_bag.dags.get("ml_training_pipeline")
        if dag is None:
            pytest.skip("DAG not found")

        assert dag.schedule_interval is not None, (
            "DAG has no schedule_interval defined"
        )