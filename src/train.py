import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import os

def main(n_estimators, max_depth):
    print("Loading data...")
    df = pd.read_csv('data/raw/creditcard.csv')
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print("Splitting data into 80% Train, 10% Val, 10% Test...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")
    
    mlflow.set_experiment("Credit_Card_Fraud_Detection")
    
    with mlflow.start_run():
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset", "CreditCardFraud")
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        print(f"Training RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        print("Evaluating model...")
        
        def log_metrics(X_data, y_true, prefix):
            y_pred = model.predict(X_data)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            
            mlflow.log_metric(f"{prefix}_accuracy", acc)
            mlflow.log_metric(f"{prefix}_f1", f1)
            mlflow.log_metric(f"{prefix}_precision", precision)
            mlflow.log_metric(f"{prefix}_recall", recall)
            
            return y_pred 

        log_metrics(X_train, y_train, "train")
        y_val_pred = log_metrics(X_val, y_val, "val")
        y_test_pred = log_metrics(X_test, y_test, "test")
        
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        cm = confusion_matrix(y_val, y_val_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Validation Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs("models", exist_ok=True)
        cm_plot_path = "models/val_confusion_matrix.png"
        plt.savefig(cm_plot_path)
        plt.close()
        
        mlflow.log_artifact(cm_plot_path)
        
        print(f"Run completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RandomForest model for Fraud Detection")
    parser.add_argument("--n_estimators", type=int, default=10, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum depth")
    args = parser.parse_args()
    
    main(args.n_estimators, args.max_depth)