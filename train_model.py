import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import joblib
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "backend", "dataset_uploads")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "backend", "voice_guardian_model.pt")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "backend", "scaler.pkl")

os.makedirs(DATASET_DIR, exist_ok=True)

class VoiceAuthenticityClassifier(nn.Module):
    def __init__(self, input_dim=13):
        super(VoiceAuthenticityClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def extract_mfcc_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) < 1600:
            return None
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        return None

def generate_synthetic_samples(num_samples=100):
    np.random.seed(42)
    synthetic_mfccs = np.random.normal(loc=1.5, scale=0.8, size=(num_samples, 13))
    authentic_mfccs = np.random.normal(loc=-0.5, scale=0.5, size=(num_samples, 13))
    return authentic_mfccs, synthetic_mfccs

def train():
    print("=== SIH26104 PyTorch Model Trainer ===")
    
    X, y = [], []
    wav_files = glob.glob(os.path.join(DATASET_DIR, "*.wav"))
    
    print(f"Found {len(wav_files)} local `.wav` dataset files.")
    
    for f in wav_files:
        feat = extract_mfcc_features(f)
        if feat is not None:
            X.append(feat)
            y.append(1.0 if "synthetic" in f.lower() else 0.0)

    auth_syn, fake_syn = generate_synthetic_samples(100)
    for feat in auth_syn:
        X.append(feat)
        y.append(0.0)
    for feat in fake_syn:
        X.append(feat)
        y.append(1.0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Feature scaler saved to {SCALER_SAVE_PATH}")

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = VoiceAuthenticityClassifier(input_dim=13)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("Training PyTorch Neural Network...")
    model.train()
    for epoch in range(1, 31):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/30] - Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model weights saved to {MODEL_SAVE_PATH}")
    print("Training Complete!")

if __name__ == "__main__":
    train()
