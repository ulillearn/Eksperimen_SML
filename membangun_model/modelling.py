import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Setup MLflow Tracking ke Localhost
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("IT_Role_Classification_Basic")

# Load Dataset (Ganti 'target_role' dengan nama kolom target yang sesuai)
df = pd.read_csv('dataset_preprocessing/cleaned_data.csv')
X = df.drop('target_role', axis=1)
y = df['target_role']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mengaktifkan Autologging Scikit-Learn
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Basic_RandomForest"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi Model Dasar: {accuracy:.4f}")