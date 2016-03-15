from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of a trained machine learning model.
    
    Args:
        model (object): The trained machine learning model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): True target variable for testing.
        
    Returns:
        dict: A dictionary containing various evaluation metrics.
    """
    print("Evaluating model performance...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=\"weighted\"),
        "recall": recall_score(y_test, y_pred, average=\"weighted\"),
        "f1_score": f1_score(y_test, y_pred, average=\"weighted\"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    # For binary classification, also calculate ROC AUC
    if len(np.unique(y_test)) == 2 and hasattr(model, \"predict_proba\"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics[\"roc_auc\"] = roc_auc_score(y_test, y_proba)
        
    print("Model evaluation complete.")
    return metrics

def print_metrics(metrics):
    """
    Prints the evaluation metrics in a readable format.
    """
    print("\n--- Model Evaluation Metrics ---")
    for metric, value in metrics.items():
        if metric == \"confusion_matrix\":
            print(f\"  {metric.replace(\"_\", \" \").title()}:\n{np.array(value)}\")
        else:
            print(f\"  {metric.replace(\"_\", \" \").title()}: {value:.4f}\")
    print("------------------------------")

if __name__ == \"__main__\":
    # Example Usage (requires dummy data and trained model from previous steps)
    import pandas as pd
    import joblib
    from data_preprocessing import load_data, preprocess_data, split_data
    from model_training import train_model
    import os

    # Ensure dummy data exists
    dummy_data = {
        \"feature_num_1\": [10, 20, 15, 25, 30, 12, 22, 18, 28, 35],
        \"feature_num_2\": [1.1, 2.2, 1.5, 2.8, 3.1, 1.3, 2.5, 1.9, 2.9, 3.8],
        \"feature_cat_1\": [\"A\", \"B\", \"A\", \"C\", \"B\", \"A\", \"C\", \"B\", \"A\", \"C\"],
        \"feature_cat_2\": [\"X\", \"Y\", \"X\", \"Z\", \"Y\", \"X\", \"Z\", \"Y\", \"X\", \"Z\"],
        \"target\": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_filepath = \"/home/ubuntu/mlops-pipeline-template/data/dummy_dataset.csv\"
    os.makedirs(os.path.dirname(dummy_filepath), exist_ok=True)
    dummy_df.to_csv(dummy_filepath, index=False)

    df = load_data(dummy_filepath)

    categorical_features = [\"feature_cat_1\", \"feature_cat_2\"]
    numerical_features = [\"feature_num_1\", \"feature_num_2\"]
    target_column = \"target\"

    X_processed, y, _ = preprocess_data(df, target_column, categorical_features, numerical_features)
    X_train, X_test, y_train, y_test = split_data(X_processed, y)

    if X_train is not None:
        # Train a Logistic Regression model
        lr_model = train_model(X_train, y_train, model_type=\"logistic_regression\")
        
        # Evaluate the model
        metrics = evaluate_model(lr_model, X_test, y_test)
        print_metrics(metrics)

        print("\nModel evaluation example finished!")
