import pandas as pd
import os
import shutil
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    print("Memulai proses re-training model...")
    
    # 1. Load Data
    data_path = "dataset_preprocessing/cleaned_data.csv"
    df = pd.read_csv(data_path)
    X = df.drop('target_role', axis=1)
    y = df['target_role']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train Model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 3. Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi Model Baru: {acc:.4f}")

    # 4. Simpan Model ke Folder untuk Docker Build
    output_dir = "model_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # Hapus direktori lama jika ada
        
    # Menyimpan model dalam format MLflow
    mlflow.sklearn.save_model(model, output_dir)
    print(f"Model berhasil disimpan di direktori {output_dir}/")

if __name__ == "__main__":
    main()