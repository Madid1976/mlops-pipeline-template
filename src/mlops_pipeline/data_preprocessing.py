import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(filepath):
    """
    Loads data from a specified CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def preprocess_data(df, target_column, categorical_features, numerical_features):
    """
    Preprocesses the input DataFrame by handling missing values, encoding categorical
    features, and scaling numerical features.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        categorical_features (list): List of categorical feature names.
        numerical_features (list): List of numerical feature names.
        
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Preprocessed features (X).
            - pd.Series: Target variable (y).
            - ColumnTransformer: The fitted preprocessor object.
    """
    if df is None:
        return None, None, None

    print("Starting data preprocessing...")

    # Separate target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify features to preprocess
    all_features = categorical_features + numerical_features
    X_processed = X[all_features]

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",  # Keep other columns not specified
    )

    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(X_processed)

    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    transformed_feature_names = numerical_features + list(cat_feature_names)

    X_final = pd.DataFrame(X_transformed, columns=transformed_feature_names, index=X.index)

    print("Data preprocessing complete.")
    return X_final, y, preprocessor

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the preprocessed data into training and testing sets.
    """
    if X is None or y is None:
        return None, None, None, None

    print(f"Splitting data into training and testing sets with test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("Data splitting complete.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example Usage (assuming a dummy CSV exists)
    # Create a dummy CSV for demonstration
    dummy_data = {
        'feature_num_1': [10, 20, 15, 25, 30, 12, 22, 18, 28, 35],
        'feature_num_2': [1.1, 2.2, 1.5, 2.8, 3.1, 1.3, 2.5, 1.9, 2.9, 3.8],
        'feature_cat_1': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'feature_cat_2': ['X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_filepath = "/home/ubuntu/mlops-pipeline-template/data/dummy_dataset.csv"
    os.makedirs(os.path.dirname(dummy_filepath), exist_ok=True)
    dummy_df.to_csv(dummy_filepath, index=False)

    df = load_data(dummy_filepath)

    if df is not None:
        categorical_features = ['feature_cat_1', 'feature_cat_2']
        numerical_features = ['feature_num_1', 'feature_num_2']
        target_column = 'target'

        X_processed, y, preprocessor = preprocess_data(df, target_column, categorical_features, numerical_features)

        if X_processed is not None and y is not None:
            X_train, X_test, y_train, y_test = split_data(X_processed, y)

            if X_train is not None:
                print("\nPreprocessing and splitting successful!")
                print(f"X_train shape: {X_train.shape}")
                print(f"y_train shape: {y_train.shape}")
                print(f"X_test shape: {X_test.shape}")
                print(f"y_test shape: {y_test.shape}")

