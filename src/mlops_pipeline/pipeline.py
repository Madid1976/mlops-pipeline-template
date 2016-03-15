import os
import pandas as pd
import joblib
from datetime import datetime

from mlops_pipeline.data_preprocessing import load_data, preprocess_data, split_data
from mlops_pipeline.model_training import train_model, save_model
from mlops_pipeline.model_evaluation import evaluate_model, print_metrics

def run_pipeline(data_filepath, target_column, categorical_features, numerical_features, model_type, model_save_path, preprocessor_save_path, test_size=0.2, random_state=42, **model_kwargs):
    """
    Runs the complete MLOps pipeline: data loading, preprocessing, training, and evaluation.
    """
    print(f"\n--- MLOps Pipeline Started at {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")} ---")

    # 1. Load Data
    df = load_data(data_filepath)
    if df is None:
        print("Pipeline aborted due to data loading error.")
        return

    # 2. Preprocess Data
    X_processed, y, preprocessor = preprocess_data(df, target_column, categorical_features, numerical_features)
    if X_processed is None or y is None:
        print("Pipeline aborted due to data preprocessing error.")
        return

    # Save preprocessor
    try:
        os.makedirs(os.path.dirname(preprocessor_save_path), exist_ok=True)
        joblib.dump(preprocessor, preprocessor_save_path)
        print(f"Preprocessor saved to {preprocessor_save_path}")
    except Exception as e:
        print(f"Error saving preprocessor: {e}")

    # 3. Split Data
    X_train, X_test, y_train, y_test = split_data(X_processed, y, test_size, random_state)
    if X_train is None:
        print("Pipeline aborted due to data splitting error.")
        return

    # 4. Train Model
    model = train_model(X_train, y_train, model_type, **model_kwargs)

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    save_model(model, model_save_path)

    # 5. Evaluate Model
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics(metrics)

    print(f"\n--- MLOps Pipeline Finished at {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")} ---")

if __name__ == "__main__":
    # Define paths and parameters
    BASE_DIR = "/home/ubuntu/mlops-pipeline-template"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    PREPROCESSORS_DIR = os.path.join(BASE_DIR, "preprocessors")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREPROCESSORS_DIR, exist_ok=True)

    data_filepath = os.path.join(DATA_DIR, "dummy_dataset.csv")
    model_save_path = os.path.join(MODELS_DIR, "trained_model.joblib")
    preprocessor_save_path = os.path.join(PREPROCESSORS_DIR, "preprocessor.joblib")

    # Create a dummy CSV for demonstration if it doesn't exist
    if not os.path.exists(data_filepath):
        dummy_data = {
            'feature_num_1': [10, 20, 15, 25, 30, 12, 22, 18, 28, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500],
            'feature_num_2': [1.1, 2.2, 1.5, 2.8, 3.1, 1.3, 2.5, 1.9, 2.9, 3.8, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0],
            'feature_cat_1': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'feature_cat_2': ['X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z'],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(data_filepath, index=False)
        print(f"Created dummy dataset at {data_filepath}")

    # Pipeline parameters
    target_column = 'target'
    categorical_features = ['feature_cat_1', 'feature_cat_2']
    numerical_features = ['feature_num_1', 'feature_num_2']
    model_type = 'random_forest'

    # Run the pipeline
    run_pipeline(data_filepath, target_column, categorical_features, numerical_features, model_type, model_save_path, preprocessor_save_path)
