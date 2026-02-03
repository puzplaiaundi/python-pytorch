from flask import Blueprint, jsonify, request
import joblib
from pathlib import Path

ia_bp = Blueprint("ia", __name__)

MODEL_PATH = Path(__file__).resolve().parents[2] / "ml" / "model.joblib"
model = joblib.load(MODEL_PATH)  # se carga una vez al arrancar

@ia_bp.route("/api/ia/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    texto = (data.get("texto") or "").strip()

    if not texto:
        return jsonify({"error": "No se ha enviado texto"}), 400

    pred = model.predict([texto])[0]

    # Probabilidades (opcional)
    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([texto])[0]
        labels = list(model.classes_)
        proba = {labels[i]: float(probs[i]) for i in range(len(labels))}

    return jsonify({
        "input": texto,
        "prediccion": pred,
        "probabilidades": proba
    })
