  
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from model import ModelBasedClassImproved, total_size, knn_classic, knn_sklearn

# Function to load and preprocess the dataset
def load_dataset(file_path):
    """
    Load the dataset from the specified file path, preprocess the data, and map labels.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
    pd.set_option('use_inf_as_na', True)  # Treat infinite values as NA
    df['Label'] = df['Label'].map({'BENIGN': 0, 'DDoS': 1})  # Map labels to numerical values
    df.dropna(inplace=True)  # Drop rows with missing values
    return df

# Main function to compare classification methods
if __name__ == "__main__":
    # Load the PortScan dataset
    df = load_dataset("../data/CICIDS_Dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN
    df.dropna(inplace=True)  # Drop rows with NaN values

    # Separate features and labels
    X = df.drop('Label', axis=1).to_numpy()
    y = df['Label'].to_numpy()

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- Method 1: KNN Classic ---
    #predictions_knn, train_time_knn, pred_time_knn, total_time_knn = knn_classic(X_train, y_train, X_test)

    # --- Method 2: KNN with scikit-learn ---
    predictions_knn_sklearn, train_time_knn_sklearn, pred_time_knn_sklearn, total_time_knn_sklearn = knn_sklearn(X_train, y_train, X_test)

    # --- Method 3: Enhanced Model-Based Class ---
    model_based_class = ModelBasedClassImproved(Lambda = 0.125)
    
    # Train the enhanced model and measure training time
    start_train_time = time.time()
    model_based_class.fit(X_train, y_train)
    end_train_time = time.time()
    train_time_model = end_train_time - start_train_time

    # Predict using the enhanced model and measure prediction time
    start_pred_time = time.time()
    predictions_model = model_based_class.predict(X_test)
    end_pred_time = time.time()
    pred_time_model = end_pred_time - start_pred_time

    total_time_model = train_time_model + pred_time_model

    # --- Metrics for each method ---
    # accuracy_knn = accuracy_score(y_test, predictions_knn)
    # recall_knn = recall_score(y_test, predictions_knn)
    # f1_knn = f1_score(y_test, predictions_knn)
    # cm_knn = confusion_matrix(y_test, predictions_knn)

    accuracy_knn_sklearn = accuracy_score(y_test, predictions_knn_sklearn)
    recall_knn_sklearn = recall_score(y_test, predictions_knn_sklearn)
    f1_knn_sklearn = f1_score(y_test, predictions_knn_sklearn)
    cm_knn_sklearn = confusion_matrix(y_test, predictions_knn_sklearn)

    accuracy_model = accuracy_score(y_test, predictions_model)
    recall_model = recall_score(y_test, predictions_model)
    f1_model = f1_score(y_test, predictions_model)
    cm_model = confusion_matrix(y_test, predictions_model)

    # --- Memory size comparison ---
    size_class_intervals = total_size(model_based_class.class_intervals)  # Memory usage for class intervals
    size_means = total_size(model_based_class.means)  # Memory usage for feature means
    size_importances = total_size(model_based_class.importances)  # Memory usage for feature importances
    total_size_model = size_class_intervals + size_means + size_importances

    # Memory size for KNN Classic and scikit-learn (entire training data is stored)
    #total_size_knn = total_size(X_train)
    total_size_knn_sklearn = total_size(X_train)

    # --- Comparative results table ["KNN Classic", "KNN scikit-learn", "New Model"] ---
    data = {
        "Model": ["KNN scikit-learn", "New Model"],
        "Accuracy": [accuracy_knn_sklearn, accuracy_model],
        "Recall": [recall_knn_sklearn, recall_model],
        "F1-Score": [f1_knn_sklearn, f1_model],
        "Training Time (s)": [train_time_knn_sklearn, train_time_model],
        "Prediction Time (s)": [pred_time_knn_sklearn, pred_time_model],
        "Data Requirement (Mo)": [total_size_knn_sklearn, total_size_model],
    }

    # # --- Comparative results table ["KNN Classic", "KNN scikit-learn", "New Model"] ---
    # data = {
    #     "Model": ["KNN Classic", "KNN scikit-learn", "New Model"],
    #     "Accuracy": [accuracy_knn, accuracy_knn_sklearn, accuracy_model],
    #     "Recall": [recall_knn, recall_knn_sklearn, recall_model],
    #     "F1-Score": [f1_knn, f1_knn_sklearn, f1_model],
    #     "Training Time (s)": [train_time_knn, train_time_knn_sklearn, train_time_model],
    #     "Prediction Time (s)": [pred_time_knn, pred_time_knn_sklearn, pred_time_model],
    #     "Data Requirement (Mo)": [total_size_knn, total_size_knn_sklearn, total_size_model],
    # }

    df_comparative = pd.DataFrame(data)  # Create a comparative table
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)


    print(df_comparative)  # Display the results
