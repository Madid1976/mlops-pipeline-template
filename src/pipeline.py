import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import os

def run_pipeline():
    # 1. Data Loading
    print("Loading data...")
    # Dummy data for demonstration
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)

    X = df[['feature1', 'feature2']]
    y = df['target']

    # 2. Data Splitting
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Model Training
    print("Training model...")
    with mlflow.start_run():
        n_estimators = 100
        max_depth = 5
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # 4. Model Evaluation
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")
        mlflow.log_metric("accuracy", accuracy)

        # 5. Model Saving
        print("Saving model...")
        mlflow.sklearn.log_model(model, "random_forest_model")

    print("Pipeline finished successfully!")

if __name__ == "__main__":
    # Ensure the src directory exists
    os.makedirs('src', exist_ok=True)
    run_pipeline()
