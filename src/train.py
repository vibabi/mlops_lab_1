import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import os

def main(n_estimators, max_depth):
    print("Loading processed data...")
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    X_train, y_train = train_df.drop('Class', axis=1), train_df['Class']
    X_val, y_val = val_df.drop('Class', axis=1), val_df['Class']
    X_test, y_test = test_df.drop('Class', axis=1), test_df['Class']
    
    mlflow.set_experiment("Credit_Card_Fraud_Detection")
    
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        print(f"Training RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, 
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        def log_metrics(X_data, y_true, prefix):
            y_pred = model.predict(X_data)
            mlflow.log_metric(f"{prefix}_accuracy", accuracy_score(y_true, y_pred))
            mlflow.log_metric(f"{prefix}_f1", f1_score(y_true, y_pred))
            mlflow.log_metric(f"{prefix}_precision", precision_score(y_true, y_pred))
            mlflow.log_metric(f"{prefix}_recall", recall_score(y_true, y_pred))
            return y_pred 

        log_metrics(X_train, y_train, "train")
        y_val_pred = log_metrics(X_val, y_val, "val")
        log_metrics(X_test, y_test, "test")
        
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        cm = confusion_matrix(y_val, y_val_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        os.makedirs("models", exist_ok=True)
        cm_plot_path = "models/val_confusion_matrix.png"
        plt.savefig(cm_plot_path)
        plt.close()
        mlflow.log_artifact(cm_plot_path)
        
        print("Run completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=10)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()
    main(args.n_estimators, args.max_depth)