import subprocess
import time
import os
from utils import setup_logging
from api import model, data_processor

logger = setup_logging()

def train_model_if_needed():
    if not os.path.exists("trained_model"):
        logger.info("No trained model found. Starting training...")
        df = data_processor.prepare_training_data()
        model.train(df)
    else:
        logger.info("Found trained model. Skipping training.")

def run_backend():
    logger.info("Starting FastAPI backend...")
    return subprocess.Popen(["uvicorn", "api:app", "--reload"])

def run_frontend():
    logger.info("Starting Streamlit frontend...")
    subprocess.Popen(["streamlit", "run", "frontend.py"])

if __name__ == "__main__":
    try:
        train_model_if_needed()
        backend_process = run_backend()
        time.sleep(5)
        run_frontend()
        backend_process.wait()
    except Exception as e:
        logger.error(f"Error starting app: {e}")