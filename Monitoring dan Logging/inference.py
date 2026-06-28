import requests
import time
import random

API_URL = "http://127.0.0.1:8000/predict"

print("Memulai simulasi pengiriman data kandidat (Tekan Ctrl+C untuk berhenti)...")

while True:
    # Membangun profil data kandidat secara acak
    payload = {
        "Education": random.choice([1, 2, 3]),
        "Python": random.choice([0, 1]),
        "SQL": random.choice([0, 1]),
        "MachineLearning": random.choice([0, 1]),
        "Java": random.choice([0, 1]),
        "AWS": random.choice([0, 1])
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            print(f"Sukses -> Prediksi: {response.json()['predicted_role']} (Konfidensi: {response.json()['confidence']:.2f})")
        else:
            print(f"Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("Gagal terhubung ke API. Pastikan prometheus_exporter.py sedang berjalan.")
    
    # Jeda acak antara 0.5 hingga 2 detik agar grafik terlihat natural
    time.sleep(random.uniform(0.5, 2.0))