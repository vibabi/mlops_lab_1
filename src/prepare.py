import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    print("Loading raw data...")
    df = pd.read_csv('data/raw/creditcard.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']

    print("Splitting data into 80% Train, 10% Val, 10% Test...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    os.makedirs('data/processed', exist_ok=True)
    
    pd.concat([X_train, y_train], axis=1).to_csv('data/processed/train.csv', index=False)
    pd.concat([X_val, y_val], axis=1).to_csv('data/processed/val.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv('data/processed/test.csv', index=False)
    
    print("Data preparation completed!")

if __name__ == "__main__":
    prepare_data()