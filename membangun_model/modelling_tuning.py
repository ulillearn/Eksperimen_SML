import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# ==========================================
# 1. SETUP DAGSHUB & MLFLOW TRACKING
# ==========================================
# Ganti dengan kredensial DagsHub Anda
REPO_OWNER = 'USERNAME_GITHUB_ANDA'
REPO_NAME = 'NAMA_REPO_DAGSHUB_ANDA'

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_experiment("IT_Role_Classification_Tuned")

# ==========================================
# 2. DATA LOADING & SPLITTING
# ==========================================
df = pd.read_csv('dataset_preprocessing/cleaned_data.csv')
X = df.drop('target_role', axis=1)
y = df['target_role']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. HYPERPARAMETER TUNING
# ==========================================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# ==========================================
# 4. MANUAL LOGGING & ARTIFACT CREATION
# ==========================================
with mlflow.start_run(run_name="RF_Tuned_Experiment"):
    print("Memulai pelatihan dan Hyperparameter Tuning...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # -- A. Manual Logging: Parameter --
    mlflow.log_params(grid_search.best_params_)
    
    # -- B. Manual Logging: Metrik Tambahan --
    # Menggunakan average='weighted' karena klasifikasi multi-kelas (lebih dari 2 peran IT)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision_weighted", precision_score(y_test, y_pred, average='weighted'))
    mlflow.log_metric("recall_weighted", recall_score(y_test, y_pred, average='weighted'))
    
    # -- C. Logging Model --
    mlflow.sklearn.log_model(best_model, "it_role_model")
    
    # -- D. Logging Artefak Visual 1: Confusion Matrix --
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - IT Role Classification')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.tight_layout()
    cm_path = 'confusion_matrix.png'
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()
    
    # -- E. Logging Artefak Visual 2: Feature Importance --
    importances = best_model.feature_importances_
    # Membatasi 10 fitur terpenting agar grafik tidak berantakan
    indices = importances.argsort()[-10:] 
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Tingkat Kepentingan')
    plt.tight_layout()
    fi_path = 'feature_importance.png'
    plt.savefig(fi_path)
    mlflow.log_artifact(fi_path)
    plt.close()

    print("Pelatihan selesai. Model, metrik, dan artefak berhasil diunggah ke DagsHub!")