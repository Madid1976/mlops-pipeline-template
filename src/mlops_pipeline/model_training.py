from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib

def train_model(X_train, y_train, model_type=\'logistic_regression\', **kwargs):
    """
    Trains a machine learning model based on the specified type.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        model_type (str): Type of model to train. Options: \'logistic_regression\', \'random_forest\', \'svm\', \'mlp\'.
        **kwargs: Additional arguments for the model constructor.
        
    Returns:
        object: The trained model.
    """
    print(f"Training {model_type} model...")
    
    if model_type == \'logistic_regression\':
        model = LogisticRegression(random_state=42, **kwargs)
    elif model_type == \'random_forest\':
        model = RandomForestClassifier(random_state=42, **kwargs)
    elif model_type == \'svm\':
        model = SVC(random_state=42, **kwargs)
    elif model_type == \'mlp\':
        model = MLPClassifier(random_state=42, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    model.fit(X_train, y_train)
    print(f"{model_type} model training complete.")
    return model

def save_model(model, filepath):
    """
    Saves the trained model to a specified file using joblib.
    """
    try:
        joblib.dump(model, filepath)
        print(f"Model saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving model to {filepath}: {e}")

if __name__ == \'__main__\':
    # Example Usage (requires dummy data from data_preprocessing.py)
    import pandas as pd
    from data_preprocessing import preprocess_data, split_data
    import os

    # Create a dummy CSV for demonstration
    dummy_data = {
        \'feature_num_1\': [10, 20, 15, 25, 30, 12, 22, 18, 28, 35],
        \'feature_num_2\': [1.1, 2.2, 1.5, 2.8, 3.1, 1.3, 2.5, 1.9, 2.9, 3.8],
        \'feature_cat_1\': [\'A\', \'B\', \'A\', \'C\', \'B\', \'A\', \'C\', \'B\', \'A\', \'C\'],
        \'feature_cat_2\': [\'X\', \'Y\', \'X\', \'Z\', \'Y\', \'X\', \'Z\', \'Y\', \'X\', \'Z\'],
        \'target\': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_filepath = "/home/ubuntu/mlops-pipeline-template/data/dummy_dataset.csv"
    os.makedirs(os.path.dirname(dummy_filepath), exist_ok=True)
    dummy_df.to_csv(dummy_filepath, index=False)

    df = pd.read_csv(dummy_filepath)

    categorical_features = [\'feature_cat_1\', \'feature_cat_2\']
    numerical_features = [\'feature_num_1\', \'feature_num_2\']
    target_column = \'target\'

    X_processed, y, _ = preprocess_data(df, target_column, categorical_features, numerical_features)
    X_train, X_test, y_train, y_test = split_data(X_processed, y)

    if X_train is not None:
        # Train a Logistic Regression model
        lr_model = train_model(X_train, y_train, model_type=\'logistic_regression\')
        save_model(lr_model, \'/home/ubuntu/mlops-pipeline-template/models/logistic_regression_model.joblib\')

        # Train a Random Forest model
        rf_model = train_model(X_train, y_train, model_type=\'random_forest\', n_estimators=50)
        save_model(rf_model, \'/home/ubuntu/mlops-pipeline-template/models/random_forest_model.joblib\')

        print("\nModel training and saving successful!")
