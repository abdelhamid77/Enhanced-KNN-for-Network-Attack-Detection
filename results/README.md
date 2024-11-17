Results Overview

This directory contains the results generated from applying the Enhanced KNN method to detect DDoS and PortScan attacks using the CICIDS 2017 dataset. The results provide insights into the performance of the proposed model compared to traditional KNN and scikit-learn KNN implementations.
Contents

    Visual Comparisons
        Comparison of Accuracy, Recall, and F1-Score for Classic KNN, scikit-learn KNN, and the New Model.png: A bar graph illustrating the accuracy, recall, and F1-score of the three models for DDoS and PortScan attacks.
        Comparison of Prediction Time for Classic KNN, scikit-learn KNN, and the New Model.png: A graph comparing the prediction times of the three models for both attack types.
        Comparison of Storage Requirements for Classic KNN, scikit-learn KNN, and the New Model.png: A graph comparing the memory usage of the three models for the datasets.

    Evaluation Metrics
        ddos_metrics.csv: A CSV file summarizing evaluation metrics for DDoS attack detection.
        portscan_metrics.csv: A CSV file summarizing evaluation metrics for PortScan attack detection.

    Detailed Results
        ddos_results.txt: A text file containing detailed experimental results for DDoS detection.
        portscan_results.txt: A text file with similar details for PortScan detection.

How to Interpret the Results

    Visual Comparisons:
        Use the graphs to quickly compare the computational efficiency, accuracy, and memory usage of the models.
    Metrics CSV Files:
        Load the CSV files into a data analysis tool to explore metrics and compare model performance in detail.
    Text Files:
        The text files contain raw experimental results, including confusion matrices, for in-depth analysis.

Future Updates

Additional results for other datasets and scenarios (e.g., infiltration attacks, financial fraud detection) will be added here as the project evolves.
