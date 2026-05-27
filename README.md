# Three-Model Deep Learning Fraud Detection System



A cutting-edge credit card fraud detection system leveraging three deep learning models: Multilayer Perceptron (MLP), Autoencoder (AE), and Variational Autoencoder (VAE). This system addresses the challenges of highly imbalanced datasets and evolving fraud patterns by combining supervised classification and unsupervised anomaly detection. Deployed via FastAPI for real-time predictions and includes an interactive dashboard.




## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [API Endpoints](#api-endpoints)
- [Usage](#usage)
- [Technologies](#technologies)
- [Contributing](#contributing)


## Features
- **Three Deep Learning Models**:
  - **MLP**: Supervised classifier for direct fraud detection.
  - **Autoencoder**: Unsupervised anomaly detection based on reconstruction errors.
  - **Variational Autoencoder**: Probabilistic model for detecting subtle fraud patterns.
- **Robust Preprocessing**: Handles imbalanced data with SMOTE and scales features using `RobustScaler`.
- **Real-Time Predictions**: FastAPI-based REST API with endpoints for individual and ensemble model predictions.
- **Interactive Dashboard**: Web interface for testing fraud detection.
- **Comprehensive Evaluation**: Metrics include Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.

## Dataset
- **Source**: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets)
- **Features**: 28 PCA-transformed anonymised features (V1–V28), transaction `Amount`, and `Class` (0 = non-fraudulent, 1 = fraudulent).
- **Imbalance**: Fraudulent transactions ≈ **0.17%**.

## Project Structure
```
Three-Model-Deep-Learning-Fraud-Detection-System/
├── app.py                    # FastAPI application
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
├── data/                     # Raw dataset (ignored in git)
├── models/                   # Trained model files (ignored)
├── saved_models/             # Model checkpoints & scaler (ignored)
├── saved_artifacts/          # Preprocessed data & thresholds (ignored)
├── figures/                  # Plots and visualizations (ignored)
├── logs/                     # Training & API logs (ignored)
├── notebooks/                # Jupyter notebooks for EDA and experiments
├── tests/                    # Unit and integration tests
└── tuner_dir/                # Hyperparameter tuning artifacts
```

## Installation

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/11snigdha11/Credit_card_fraud_dl
   ```

2. **Create & Activate Virtual Environment**:
   ```bash
   python -m venv fraud-dl-env
   ```
   - On Windows (PowerShell):
     ```bash
     fraud-dl-env\Scripts\Activate.ps1
     ```
   - On macOS/Linux:
     ```bash
     source fraud-dl-env/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI Application**:
   ```bash
   uvicorn app:app --reload
   ```

5. **Access the API**:
   - **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - **Dashboard**: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)

## API Endpoints
| Endpoint                | Method | Description                              |
|-------------------------|--------|------------------------------------------|
| `/`                     | GET    | Root welcome message                    |
| `/health`               | GET    | API health check                        |
| `/models`               | GET    | List all available models                |
| `/predict/mlp`          | POST   | Fraud detection using MLP               |
| `/predict/autoencoder`  | POST   | Fraud detection using Autoencoder       |
| `/predict/vae`          | POST   | Fraud detection using VAE               |
| `/predict/ensemble`     | POST   | Fraud detection using ensemble voting   |
| `/dashboard`            | GET    | Interactive dashboard for predictions   |

## Usage
1. **Preprocessing**: The system handles duplicates, missing values, and scales data using `RobustScaler`. SMOTE is applied to address class imbalance.
2. **Training**: Models are implemented in PyTorch with dropout, batch normalisation, and early stopping.
3. **Evaluation**: Models are evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.
4. **Testing**: Use the Swagger UI or cURL to send sample transactions for real-time predictions.

## Technologies
- **Language**: Python
- **Libraries**:
  - Data: `pandas`, `numpy`
  - Visualisation: `matplotlib`, `seaborn`
  - Deep Learning: `torch` (PyTorch)
  - Model Persistence: `joblib`, `pickle`
- **Deployment**: FastAPI
- **Version Control**: Git + GitHub


