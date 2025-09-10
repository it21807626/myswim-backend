import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

# -----------------------------
# No legacy flags/shims needed!
# This env is TF 2.11 + tf.keras.
# -----------------------------

app = Flask(__name__)
CORS(app)

# Manifest
import json
with open(os.path.join("config", "ensemble_manifest.json"), "r") as f:
    MANIFEST = json.load(f)

def _get_model_path(kind: str):
    for m in MANIFEST.get("models", []):
        if m.get("kind") == kind:
            return m.get("path")
    raise RuntimeError(f"No model path found for kind='{kind}'")

M1D_PATH = _get_model_path("1d")   # e.g. "models/cnn1d_anysafe_balanced.h5"
M2D_PATH = _get_model_path("2d")   # e.g. "models/cnn2d_anysafe_improved.h5"

WINDOW  = int(MANIFEST.get("window", 30))
STRIDE  = int(MANIFEST.get("stride", 15))
ENSEMBLE_CFG = MANIFEST.get("ensemble", {})
AGGREGATE = ENSEMBLE_CFG.get("aggregate", MANIFEST.get("aggregate", "max")).lower()

# ---- Load models (plain .h5) ----
MODEL_1D = tf.keras.models.load_model(M1D_PATH, compile=False)
MODEL_2D = tf.keras.models.load_model(M2D_PATH, compile=False)

ENSEMBLE_MODEL = None
if ENSEMBLE_CFG.get("kind") == "model":
    ens_path = ENSEMBLE_CFG.get("path")
    if ens_path and os.path.exists(ens_path):
        ENSEMBLE_MODEL = tf.keras.models.load_model(ens_path, compile=False)

# ---- Load normalization stats ----
norm1d = np.load(os.path.join("norm", "cnn1d_data_anysafe.npz"))
norm2d = np.load(os.path.join("norm", "cnn2d_data_anysafe.npz"))
MU1D, SIG1D = norm1d["mu"], norm1d["sigma"]
MU2D, SIG2D = norm2d["mu"], norm2d["sigma"]

def normalize(arr, mu, sigma, eps=1e-6):
    return (arr - mu) / (sigma + eps)

def _to_logits(p, eps=1e-7):
    p = np.clip(p, eps, 1. - eps)
    return np.log(p/(1.-p))

def combine_window_probs(p1, p2):
    if ENSEMBLE_CFG.get("kind") == "model" and ENSEMBLE_MODEL is not None:
        X = np.stack([p1, p2], axis=1) if ENSEMBLE_CFG.get("input","probs")!="logits" \
            else np.stack([_to_logits(p1), _to_logits(p2)], axis=1)
        y = ENSEMBLE_MODEL.predict(X, verbose=0).squeeze()
        return np.atleast_1d(y).astype(float)
    # default average
    return (0.5*p1 + 0.5*p2).astype(float)

def aggregate_clip(win_probs):
    return float(np.mean(win_probs) if AGGREGATE=="mean" else np.max(win_probs))

def get_threshold():
    th = ENSEMBLE_CFG.get("threshold", MANIFEST.get("threshold_risky", 0.463))
    return float(th)

@app.get("/")
def root():
    return jsonify({"message": "myswim backend up", "tf": tf.__version__})

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "window": WINDOW, "stride": STRIDE,
        "aggregate": AGGREGATE,
        "ensemble_model_loaded": ENSEMBLE_MODEL is not None,
        "threshold": get_threshold()
    })

# Keep your /predict JSON + multipart(video) endpoint unchanged below.
# Your existing pose-extraction, windowize, build 2D, normalize,
# predict with MODEL_1D/MODEL_2D → ensemble → aggregate → return JSON.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

