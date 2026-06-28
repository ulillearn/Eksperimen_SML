import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def preprocess_data(df):
    print("Memulai proses preprocessing...")

    # 1. Drop kolom tidak relevan
    df_clean = df.drop(columns=['Name'])

    # 2. Encode JobRole
    label_encoder = LabelEncoder()
    df_clean['JobRole_label'] = label_encoder.fit_transform(df_clean['JobRole'])

    # 3. One-hot encode Education
    df_clean = pd.get_dummies(df_clean, columns=['Education'])

    # 4. Scale YearsExperience
    scaler = StandardScaler()
    df_clean['YearsExperience_scaled'] = scaler.fit_transform(df_clean[['YearsExperience']])

    # 5. TF-IDF encoding Skills
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: [s.strip() for s in x.split(',')],
        lowercase=False
    )
    skills_tfidf = vectorizer.fit_transform(df_clean['Skills'].astype(str))
    skills_df = pd.DataFrame(skills_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    df_clean = pd.concat([df_clean.reset_index(drop=True), skills_df], axis=1)

    # 6. Drop duplikat
    df_clean = df_clean.drop_duplicates()

    print("Preprocessing selesai.")
    return df_clean

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data bersih disimpan ke {output_path}")

# Fungsi utama untuk menjalankan semua langkah preprocessing
if __name__ == "__main__":
    INPUT_PATH = "dataset_raw/resume_dataset.csv"
    OUTPUT_PATH = "preprocessing/dataset_preprocessing/cleaned_data.csv"

    raw_df = load_data(INPUT_PATH)
    clean_df = preprocess_data(raw_df)
    save_data(clean_df, OUTPUT_PATH)