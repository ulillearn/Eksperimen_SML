import time
import psutil
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Gauge, Histogram

app = FastAPI(title="IT Role Classification API")

# ==========================================
# DEFINISI 10 METRIK PROMETHEUS (Target Advanced)
# ==========================================
REQUEST_COUNT = Counter('api_requests_total', 'Total prediksi yang diminta')
ERROR_COUNT = Counter('api_errors_total', 'Total error saat prediksi')
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total prediksi per kelas', ['role'])
LATENCY = Histogram('api_latency_seconds', 'Waktu respons API')
CPU_USAGE = Gauge('system_cpu_usage_percent', 'Penggunaan CPU oleh sistem')
MEMORY_USAGE = Gauge('system_memory_usage_percent', 'Penggunaan Memori oleh sistem')
ACTIVE_REQUESTS = Gauge('api_active_requests', 'Jumlah request yang sedang diproses')
MODEL_VERSION = Gauge('model_version_info', 'Versi model yang berjalan')
CONFIDENCE_SCORE = Histogram('model_confidence_score', 'Skor probabilitas prediksi')
DATA_DRIFT_SCORE = Gauge('data_drift_indicator', 'Simulasi metrik data drift')

# Tambahkan endpoint /metrics untuk Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Load Model dari hasil eksperimen MLflow
# (Pastikan path mengarah ke model yang sudah di-training/di-download)
try:
    MODEL_PATH = "../Membangun_model/mlruns/0/LOGGED_MODEL_ID/artifacts/it_role_model" 
    # Sesuaikan path di atas saat eksekusi nanti
    model = mlflow.sklearn.load_model(MODEL_PATH)
    MODEL_VERSION.set(1.0)
except Exception as e:
    print("Model belum diload. Harap sesuaikan MODEL_PATH saat debugging.")
    model = None

class CandidateData(BaseModel):
    Education: int
    Python: int
    SQL: int
    MachineLearning: int
    Java: int
    AWS: int

@app.get("/")
def health_check():
    return {"status": "Model API is running"}

@app.post("/predict")
def predict_role(data: CandidateData):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    # Update System Metrics
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    
    try:
        # Simulasi Data Drift (acak antara 0.0 - 0.2 untuk kebutuhan monitoring)
        DATA_DRIFT_SCORE.set(psutil.cpu_percent() / 500) 
        
        # Konversi input ke DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Prediksi
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        max_prob = max(probabilities)
        
        # Catat Metrik
        PREDICTION_COUNTER.labels(role=prediction).inc()
        CONFIDENCE_SCORE.observe(max_prob)
        
        process_time = time.time() - start_time
        LATENCY.observe(process_time)
        ACTIVE_REQUESTS.dec()
        
        return {"predicted_role": prediction, "confidence": max_prob}
        
    except Exception as e:
        ERROR_COUNT.inc()
        ACTIVE_REQUESTS.dec()
        raise HTTPException(status_code=500, detail=str(e))

# Cara menjalankan: uvicorn 3.prometheus_exporter:app --host 0.0.0.0 --port 8000