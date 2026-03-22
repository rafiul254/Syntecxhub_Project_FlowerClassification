import os
import csv
import io
import joblib
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

MODEL_DIR    = "outputs/models"
TARGET_NAMES = ["setosa", "versicolor", "virginica"]
MAX_HISTORY  = 20

prediction_history = []

def load_all_models():
    required = [
        "logistic_regression.pkl",
        "decision_tree.pkl",
        "random_forest.pkl",
        "svm.pkl",
        "scaler.pkl",
        "metadata.pkl",
    ]
    for f in required:
        path = os.path.join(MODEL_DIR, f)
        if not os.path.exists(path):
            print(f"[ERROR] Missing: {path}")
            print("        Run: python model/train_model.py")
            return None

    models = {
        "lr":     joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
        "dt":     joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl")),
        "rf":     joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
        "svm":    joblib.load(os.path.join(MODEL_DIR, "svm.pkl")),
        "scaler": joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
        "meta":   joblib.load(os.path.join(MODEL_DIR, "metadata.pkl")),
    }
    print("[OK] All 4 models loaded successfully.")
    return models


MODELS = load_all_models()

def predict_model(model, features, scaled=False):
    X = MODELS["scaler"].transform(features) if scaled else features
    idx   = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    return {
        "species":    TARGET_NAMES[idx],
        "confidence": round(float(max(proba)) * 100, 2),
        "probabilities": {
            TARGET_NAMES[i]: round(proba[i] * 100, 2)
            for i in range(3)
        }
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/charts")
def charts():
    meta = MODELS["meta"] if MODELS else {}
    return render_template("charts.html", meta=meta)


@app.route("/predict", methods=["POST"])
def predict():
    if MODELS is None:
        return jsonify({"error": "Models not loaded. Run train_model.py first."}), 500

    data = request.get_json()
    try:
        features = np.array([[
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"]),
        ]])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    results = {
        "logistic_regression": predict_model(MODELS["lr"],  features, scaled=True),
        "decision_tree":       predict_model(MODELS["dt"],  features, scaled=False),
        "random_forest":       predict_model(MODELS["rf"],  features, scaled=False),
        "svm":                 predict_model(MODELS["svm"], features, scaled=True),
    }

    final = results["random_forest"]["species"]

    votes      = [r["species"] for r in results.values()]
    vote_count = votes.count(final)
    agreement  = f"{vote_count}/4 models agree"

    entry = {
        "id":         len(prediction_history) + 1,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time":       datetime.now().strftime("%H:%M:%S"),
        "inputs": {
            "sepal_length": float(data["sepal_length"]),
            "sepal_width":  float(data["sepal_width"]),
            "petal_length": float(data["petal_length"]),
            "petal_width":  float(data["petal_width"]),
        },
        "prediction": final,
        "confidence": results["random_forest"]["confidence"],
        "agreement":  agreement,
    }
    prediction_history.append(entry)
    if len(prediction_history) > MAX_HISTORY:
        prediction_history.pop(0)

    return jsonify({
        "results":          results,
        "final_prediction": final,
        "agreement":        agreement,
        "vote_count":       vote_count,
        "history_count":    len(prediction_history),
    })


@app.route("/history")
def history():
    return jsonify(list(reversed(prediction_history)))


@app.route("/history/clear", methods=["POST"])
def clear_history():
    prediction_history.clear()
    return jsonify({"status": "cleared"})


@app.route("/export")
def export():
    if not prediction_history:
        return jsonify({"error": "No predictions to export yet."}), 400

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "#", "Timestamp",
        "Sepal Length (cm)", "Sepal Width (cm)",
        "Petal Length (cm)", "Petal Width (cm)",
        "Prediction", "Confidence (%)", "Model Agreement"
    ])
    for h in prediction_history:
        inp = h["inputs"]
        writer.writerow([
            h["id"], h["timestamp"],
            inp["sepal_length"], inp["sepal_width"],
            inp["petal_length"], inp["petal_width"],
            h["prediction"], h["confidence"], h["agreement"],
        ])
    output.seek(0)
    filename = f"iris_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "models_loaded": MODELS is not None,
        "history_count": len(prediction_history),
    })

if __name__ == "__main__":
    print("\n  Iris Flower Classifier — Flask Server")
    print("  Open → http://localhost:5000\n")
    app.run(debug=True, port=5000)
