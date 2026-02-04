import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import joblib

# --------------------
# Configuración
# --------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "model_nn.pt"
VECTORIZER_PATH = BASE_DIR / "vectorizer.joblib"
LABELS_PATH = BASE_DIR / "labels.joblib"

EPOCHS = 30
BATCH_SIZE = 4
HIDDEN_DIM = 32
LR = 0.01

# --------------------
# Red neuronal
# --------------------
class TextNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# --------------------
# Entrenamiento
# --------------------
def main():
    print("📄 Cargando dataset...")
    df = pd.read_csv(DATASET_PATH)

    X_text = df["texto"].astype(str)
    y_text = df["label"].astype(str)

    print("🔤 Vectorizando texto...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="spanish",
        min_df=1
    )
    X = vectorizer.fit_transform(X_text).toarray()

    print("🏷️ Codificando etiquetas...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    # Guardamos vectorizador y etiquetas
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABELS_PATH)

    # Tensores
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Modelo
    model = TextNN(
        input_dim=X.shape[1],
        hidden_dim=HIDDEN_DIM,
        output_dim=len(label_encoder.classes_)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("🧠 Entrenando red neuronal...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    # Guardar modelo
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Modelo neuronal guardado en: {MODEL_PATH}")
    print("🎉 Entrenamiento finalizado")

if __name__ == "__main__":
    main()
