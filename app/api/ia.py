from flask import Blueprint, request, jsonify
import torch
import torch.nn as nn
import joblib
from pathlib import Path
import numpy as np

ia_bp = Blueprint("ia", __name__)

# --------------------
# Rutas
# --------------------
BASE_DIR = Path(__file__).resolve().parents[2]
ML_DIR = BASE_DIR / "ml"

MODEL_PATH = ML_DIR / "model_nn.pt"
VECTORIZER_PATH = ML_DIR / "vectorizer.joblib"
LABELS_PATH = ML_DIR / "labels.joblib"

# --------------------
# Red neuronal (MISMA que en train_nn.py)
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
# Carga de recursos (una sola vez)
# --------------------
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(LABELS_PATH)

input_dim = len(vectorizer.get_feature_names_out())
output_dim = len(label_encoder.classes_)

model = TextNN(
    input_dim=input_dim,
    hidden_dim=32,
    output_dim=output_dim
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()  # modo inferencia

# --------------------
# Endpoints
# --------------------
@ia_bp.route("/api/ia/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "ok",
        "modelo": "red_neuronal_pytorch"
    })


@ia_bp.route("/api/ia/predict", methods=["POST"])
def predict():
    data = request.get_json()
    texto = data.get("texto", "")

    if not texto.strip():
        return jsonify({"error": "Texto vacío"}), 400

    # Vectorizar
    X = vectorizer.transform([texto]).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Inferencia
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    # Preparar respuesta
    clases = label_encoder.classes_
    prob_dict = {
        clases[i]: float(probs[i])
        for i in range(len(clases))
    }

    max_idx = int(np.argmax(probs))
    prediccion = clases[max_idx]
    confianza = float(probs[max_idx])

    return jsonify({
        "input": texto,
        "prediccion": prediccion,
        "confianza": confianza,
        "probabilidades": prob_dict
    })
