import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from model import ModelBasedClassImproved, total_size, knn_classic, knn_sklearn

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    pd.set_option('use_inf_as_na', True)
    df['Label'] = df['Label'].map({'BENIGN': 0, 'PortScan': 1})
    df.dropna(inplace=True)
    return df

# Comparaison des méthodes
if __name__ == "__main__":
    df = load_dataset("CICIDS Dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.drop('Label', axis=1).to_numpy()
    y = df['Label'].to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # KNN classique sans sklearn
    predictions_knn, train_time_knn, pred_time_knn, total_time_knn = knn_classic(X_train, y_train, X_test)

    # KNN avec sklearn
    predictions_knn_sklearn, train_time_knn_sklearn, pred_time_knn_sklearn, total_time_knn_sklearn = knn_sklearn(X_train, y_train, X_test)

    # Votre modèle amélioré
    model_based_class = ModelBasedClassImproved(weighted=False)
    
    start_train_time = time.time()
    model_based_class.fit(X_train, y_train)
    end_train_time = time.time()
    train_time_model = end_train_time - start_train_time

    start_pred_time = time.time()
    predictions_model = model_based_class.predict(X_test)
    end_pred_time = time.time()
    pred_time_model = end_pred_time - start_pred_time

    total_time_model = train_time_model + pred_time_model

    # Métriques pour chaque méthode
    accuracy_knn = accuracy_score(y_test, predictions_knn)
    recall_knn = recall_score(y_test, predictions_knn)
    f1_knn = f1_score(y_test, predictions_knn)
    cm_knn = confusion_matrix(y_test, predictions_knn)

    accuracy_knn_sklearn = accuracy_score(y_test, predictions_knn_sklearn)
    recall_knn_sklearn = recall_score(y_test, predictions_knn_sklearn)
    f1_knn_sklearn = f1_score(y_test, predictions_knn_sklearn)
    cm_knn_sklearn = confusion_matrix(y_test, predictions_knn_sklearn)

    accuracy_model = accuracy_score(y_test, predictions_model)
    recall_model = recall_score(y_test, predictions_model)
    f1_model = f1_score(y_test, predictions_model)
    cm_model = confusion_matrix(y_test, predictions_model)

    # Comparaison de la taille des objets en mémoire
    size_class_intervals = total_size(model_based_class.class_intervals)
    size_means = total_size(model_based_class.means)
    size_importances = total_size(model_based_class.importances)
    total_size_model = size_class_intervals + size_means + size_importances

    # Taille mémoire des données (données stockées pour faire la prédiction)
    total_size_knn = total_size(X_train)
    total_size_knn_sklearn = total_size(X_train)

    # Tableau comparatif
    data = {
        "Model": ["KNN Classic", "KNN scikit-learn", "New Model"],
        "Accuracy": [accuracy_knn, accuracy_knn_sklearn, accuracy_model],
        "Recall": [recall_knn, recall_knn_sklearn, recall_model],
        "F1-Score": [f1_knn, f1_knn_sklearn, f1_model],
        "Training Time (s)": [train_time_knn, train_time_knn_sklearn, train_time_model],
        "Prediction Time (s)": [pred_time_knn, pred_time_knn_sklearn, pred_time_model],
        "Data Requirement (Mo)": [total_size_knn, total_size_knn_sklearn, total_size_model],
        "Confusion Matrix": [cm_knn, cm_knn_sklearn, cm_model]
    }

    df_comparative = pd.DataFrame(data)
    print(df_comparative)
