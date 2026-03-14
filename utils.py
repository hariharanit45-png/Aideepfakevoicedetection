import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import glob

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        # pitch = np.mean(librosa.yin(y, fmin=75, fmax=300, sr=sr))
        # jitter = np.std(librosa.yin(y, fmin=75, fmax=300, sr=sr))
        pitch = 0  # placeholder
        jitter = 0  # placeholder
        energy = np.mean(librosa.feature.rms(y=y))
        features = np.concatenate([mfccs_mean, [pitch, jitter, energy]])
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def create_dataset(real_csv, fake_dir, output_csv):
    # Load real data
    df_real = pd.read_csv(real_csv)
    data = []
    for _, row in df_real.iterrows():
        features = [row[f'mfcc_{i}'] for i in range(13)] + [row['pitch'], row['jitter'], row['energy']]
        data.append(features + [1])  # label 1 for real

    # Load fake data
    fake_files = glob.glob(os.path.join(fake_dir, '*.mp3'))
    for file in fake_files:
        features = extract_features(file)
        if features is not None:
            data.append(list(features) + [0])  # label 0 for fake

    # Create DataFrame
    columns = [f'mfcc_{i}' for i in range(13)] + ['pitch', 'jitter', 'energy', 'label']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    return df

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    print(classification_report(y_test, y_pred))
    return model

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def predict_audio(model, audio_path):
    features = extract_features(audio_path)
    if features is not None:
        pred = model.predict([features])
        return 'Real' if pred[0] == 1 else 'Fake'
    return 'Error'