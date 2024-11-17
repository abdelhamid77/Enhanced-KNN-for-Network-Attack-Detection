Instructions for Preparing the Dataset

    Download the CICIDS 2017 Dataset
        Visit the official Canadian Institute for Cybersecurity (CIC) dataset page to download the CICIDS 2017 dataset.
        Download the required CSV files for DDoS and PortScan detection.

    Organize the Files
        Create a folder named CICIDS_Dataset within the data directory of this project.
        Place the downloaded CSV files into the CICIDS_Dataset directory.

    File Structure
    After downloading, the structure should look like this:

data/
├── CICIDS_Dataset/
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── README.md 

Preprocessing Note

    Ensure that the files are clean and free of missing or infinite values.
    The scripts in the src directory are designed to handle the dataset in this structure.

Dataset License

    Ensure compliance with the dataset’s license terms as provided by the Canadian Institute for Cybersecurity.
