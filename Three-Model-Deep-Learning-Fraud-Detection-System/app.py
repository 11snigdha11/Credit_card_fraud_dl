#!/usr/bin/env python3
"""
Three-Model Fraud Detection FastAPI Application
Author: Snigdha
GitHub: https://github.com/11snigdha11/Credit_card_fraud_dl

Description:
    Welcome to the Fraud Detection API — a high-performance, multi-model
    deep learning service for detecting fraudulent transactions in real-time.

    This API deploys three complementary models:
        1. MLP (Multilayer Perceptron) - supervised fraud classifier
        2. Autoencoder - unsupervised anomaly detector
        3. Variational Autoencoder (VAE) - advanced unsupervised anomaly detector

    Key Features:
        - Individual endpoints for each model and an ensemble endpoint
        - Automatic preprocessing of transaction data
        - Threshold-based anomaly detection for robust fraud identification
        - Fully logged and ready for production deployment
        - Minimal HTML dashboard for interactive predictions
"""

import os
import json
import joblib
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch

# Import models
from models.mlp import MLP
from models.autoencoder import Autoencoder
from models.vae import VAE

# -------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------
app = FastAPI(
    title="Three-Model Fraud Detection API",
    version="1.0",
    description="""
A multi-model deep learning service for detecting fraudulent transactions in real-time.

## Models Available:
1. **MLP** - Supervised fraud classifier
2. **Autoencoder** - Unsupervised anomaly detector
3. **VAE** - Advanced unsupervised anomaly detector

## Key Features:
- Individual endpoints for each model and an ensemble endpoint
- Automatic preprocessing of transaction data
- Threshold-based anomaly detection for robust fraud identification
- Fully logged and ready for production deployment
- Minimal HTML dashboard for interactive predictions

**Author:** Nafisa Lawal Idris  
**GitHub:** [https://github.com/nafisalawalidris](https://github.com/nafisalawalidris)
""",
    contact={
        "name": "Nafisa Lawal Idris",
        "url": "https://nafisalawalidris.github.io/13/"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
ARTIFACTS_DIR = "saved_artifacts"
MODELS_DIR = "saved_models"

SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
MLP_PATH = os.path.join(MODELS_DIR, "mlp_model.pth")
AE_PATH = os.path.join(MODELS_DIR, "autoencoder_model.pth")
VAE_PATH = os.path.join(MODELS_DIR, "vae_model.pth")

THRESHOLDS_JSON = os.path.join(ARTIFACTS_DIR, "thresholds.json")
AE_THRESH_PATH = os.path.join(ARTIFACTS_DIR, "threshold_ae.pkl")
VAE_THRESH_PATH = os.path.join(ARTIFACTS_DIR, "threshold_vae.pkl")

# -------------------------------------------------------------------
# Load Models and Artifacts
# -------------------------------------------------------------------
logger.info("Loading models and artifacts for fraud detection system...")

# Scaler
scaler = joblib.load(SCALER_PATH)
logger.info("✓ Scaler loaded successfully")

# Thresholds
thresholds = {}
if os.path.exists(THRESHOLDS_JSON):
    with open(THRESHOLDS_JSON, "r") as f:
        thresholds = json.load(f)
    logger.info("✓ Thresholds loaded from JSON")
else:
    thresholds["autoencoder"] = float(joblib.load(AE_THRESH_PATH))
    thresholds["vae"] = float(joblib.load(VAE_THRESH_PATH))
    logger.info("✓ Thresholds loaded from individual .pkl files")

# MLP
mlp_model = MLP(input_dim=30)
mlp_model.load_state_dict(torch.load(MLP_PATH, map_location="cpu"))
mlp_model.eval()
logger.info("✓ MLP model loaded successfully (state_dict)")

# Autoencoder
ae_model = Autoencoder(input_dim=30)
ae_model.load_state_dict(torch.load(AE_PATH, map_location="cpu"))
ae_model.eval()
logger.info("✓ Autoencoder model loaded successfully (state_dict)")

# VAE
vae_model = VAE(input_dim=30, latent_dim=8)
vae_model.load_state_dict(torch.load(VAE_PATH, map_location="cpu"))
vae_model.eval()
logger.info("✓ VAE model loaded successfully (state_dict)")

# -------------------------------------------------------------------
# Request Schema
# -------------------------------------------------------------------
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
FEATURE_ORDER = [
    "Time", "V1", "V2", "V3", "V4", "V5",
    "V6", "V7", "V8", "V9", "V10", "V11",
    "V12", "V13", "V14", "V15", "V16", "V17",
    "V18", "V19", "V20", "V21", "V22", "V23",
    "V24", "V25", "V26", "V27", "V28", "Amount"
]

def preprocess(transaction: Transaction):
    """Convert request to scaled numpy array with correct feature order"""
    features = np.array([[getattr(transaction, f) for f in FEATURE_ORDER]])
    return scaler.transform(features)

def predict_mlp(x_scaled: np.ndarray):
    with torch.no_grad():
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        pred = mlp_model(x_tensor).item()
    return pred, pred > 0.5

def predict_autoencoder(x_scaled: np.ndarray):
    with torch.no_grad():
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        recon = ae_model(x_tensor)
        error = torch.mean((x_tensor - recon) ** 2, dim=1).item()
    return error, error > thresholds["autoencoder"]

def predict_vae(x_scaled: np.ndarray):
    with torch.no_grad():
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        recon, _, _ = vae_model(x_tensor)
        error = torch.mean((x_tensor - recon) ** 2, dim=1).item()
    return error, error > thresholds["vae"]

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/", summary="API Root", tags=["General"])
def root():
    """Welcome message for the Fraud Detection API."""
    return {"message": "Fraud Detection API is running! Visit /docs for Swagger UI or /dashboard for interactive testing."}

@app.get("/health", summary="Health Check", tags=["General"])
def health_check():
    """Check if the API is running and healthy."""
    return {"status": "healthy"}

@app.get("/models", summary="List Models", tags=["Models"])
def list_models():
    """List all available models."""
    return {"models": ["mlp", "autoencoder", "vae", "ensemble"]}

@app.post("/predict/mlp", summary="Predict with MLP", tags=["Predictions"])
def predict_with_mlp(transaction: Transaction):
    """Make a fraud prediction using the MLP model."""
    try:
        x_scaled = preprocess(transaction)
        prob, is_fraud = predict_mlp(x_scaled)
        return {"model": "MLP", "probability": prob, "is_fraud": bool(is_fraud)}
    except Exception as e:
        logger.error(f"MLP prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/autoencoder", summary="Predict with Autoencoder", tags=["Predictions"])
def predict_with_autoencoder(transaction: Transaction):
    """Make a fraud prediction using the Autoencoder model."""
    try:
        x_scaled = preprocess(transaction)
        error, is_fraud = predict_autoencoder(x_scaled)
        return {"model": "Autoencoder", "anomaly_score": error, "is_fraud": bool(is_fraud)}
    except Exception as e:
        logger.error(f"Autoencoder prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/vae", summary="Predict with VAE", tags=["Predictions"])
def predict_with_vae(transaction: Transaction):
    """Make a fraud prediction using the VAE model."""
    try:
        x_scaled = preprocess(transaction)
        error, is_fraud = predict_vae(x_scaled)
        return {"model": "VAE", "anomaly_score": error, "is_fraud": bool(is_fraud)}
    except Exception as e:
        logger.error(f"VAE prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/ensemble", summary="Predict with Ensemble", tags=["Predictions"])
def predict_ensemble(transaction: Transaction):
    """Make a fraud prediction using the ensemble of all models."""
    try:
        x_scaled = preprocess(transaction)
        mlp_prob, mlp_fraud = predict_mlp(x_scaled)
        ae_error, ae_fraud = predict_autoencoder(x_scaled)
        vae_error, vae_fraud = predict_vae(x_scaled)

        votes = [mlp_fraud, ae_fraud, vae_fraud]
        ensemble_fraud = sum(votes) >= 2

        return {
            "model": "Ensemble",
            "ensemble_is_fraud": bool(ensemble_fraud),
            "mlp_prediction": mlp_prob,
            "mlp_is_fraud": bool(mlp_fraud),
            "ae_anomaly_score": ae_error,
            "ae_is_fraud": bool(ae_fraud),
            "vae_anomaly_score": vae_error,
            "vae_is_fraud": bool(vae_fraud),
        }
    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Minimal HTML Dashboard
# -------------------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse, summary="Interactive Dashboard", tags=["Dashboard"])
def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; }
            input { width: 60px; }
            button { margin-top: 10px; padding: 8px 12px; }
            #result { margin-top: 20px; font-weight: bold; color: #e74c3c; }
        </style>
    </head>
    <body>
        <h1>Fraud Detection Dashboard</h1>
        <p>Fill transaction features and predict fraud using ensemble model.</p>
        <form id="fraudForm">
            Time: <input name="Time" value="0"><br>
            V1: <input name="V1" value="0">
            V2: <input name="V2" value="0">
            V3: <input name="V3" value="0"><br>
            V4: <input name="V4" value="0">
            V5: <input name="V5" value="0">
            V6: <input name="V6" value="0"><br>
            Amount: <input name="Amount" value="0"><br>
            <button type="button" onclick="predict()">Predict Fraud</button>
        </form>
        <div id="result"></div>
        <script>
            async function predict() {
                const form = document.getElementById('fraudForm');
                const data = {};
                for(let el of form.elements) {
                    if(el.name) data[el.name] = parseFloat(el.value);
                }
                // Fill remaining features with 0
                const features = ["V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28"];
                features.forEach(f => data[f]=0);

                const response = await fetch('/predict/ensemble', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                document.getElementById('result').innerText = JSON.stringify(result, null, 2);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
    
# -------------------------------------------------------------------
# Main Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Fraud Detection API...")
    logger.info("Use: uvicorn app:app --host 0.0.0.0 --port 8000 --reload")
    logger.info("Or visit: http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
