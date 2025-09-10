import os, json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras as K

# ---- Patch legacy InputLayer configs (batch_shape -> batch_input_shape)
class PatchedInputLayer(K.layers.InputLayer):
    @classmethod
    def from_config(cls, config):
        if "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        return super().from_config(config)

CUSTOMS = {"InputLayer": PatchedInputLayer}

app = Flask(__name__)
CORS(app)

# ---------- Config & manifest ----------
with open(os.path.join("config", "ensemble_manifest.json"), "r") as f:
    MANIFEST = json.load(f)

def _get_model_path(kind: str):
    for m in MANIFEST.get("models", []):
        if m.get("kind") == kind:
            return m.get("path")
    raise RuntimeError(f"No model path found for kind='{kind}'")

M1D_PATH = _get_model_path("1d")
M2D_PATH = _get_model_path("2d")
ENSEMBLE_CFG = MANIFEST.get("ensemble", {})
AGGREGATE = ENSEMBLE_CFG.get("aggregate", MANIFEST.get("aggregate", "max")).lower()
WINDOW  = int(MANIFEST.get("window", 30))
STRIDE  = int(MANIFEST.get("stride", 15))

# ---------- Lazy model holders ----------
MODEL_1D = None
MODEL_2D = None
ENSEMBLE_MODEL = None
_LAST_LOAD_ERR = None

def _safe_load_models():
    """Load models once; store any error to surface via /health & /predict."""
    global MODEL_1D, MODEL_2D, ENSEMBLE_MODEL, _LAST_LOAD_ERR
    if MODEL_1D is not None and MODEL_2D is not None:
        return True

    try:
        MODEL_1D = tf.keras.models.load_model(M1D_PATH, compile=False, custom_objects=CUSTOMS)
        MODEL_2D = tf.keras.models.load_model(M2D_PATH, compile=False, custom_objects=CUSTOMS)
        # Optional ensemble
        if ENSEMBLE_CFG.get("kind") == "model":
            ens_path = ENSEMBLE_CFG.get("path")
            if ens_path and os.path.exists(ens_path):
                ENSEMBLE_MODEL = tf.keras.models.load_model(ens_path, compile=False, custom_objects=CUSTOMS)
        _LAST_LOAD_ERR = None
        return True
    except Exception as e:
        _LAST_LOAD_ERR = f"{type(e).__name__}: {e}"
        # keep models as None so we can retry on next request
        MODEL_1D = None
        MODEL_2D = None
        ENSEMBLE_MODEL = None
        return False

# ---------- Norm stats ----------
def _load_norm():
    n1 = np.load(os.path.join("norm", "cnn1d_data_anysafe.npz"))
    n2 = np.load(os.path.join("norm", "cnn2d_data_anysafe.npz"))
    return (n1["mu"], n1["sigma"]), (n2["mu"], n2["sigma"])

(MU1D, SIG1D), (MU2D, SIG2D) = _load_norm()

def normalize(arr, mu, sigma, eps=1e-6):
    return (arr - mu) / (sigma + eps)

def _to_logits(p, eps=1e-7):
    p = np.clip(p, eps, 1. - eps)
    return np.log(p/(1.-p))

def combine_window_probs(p1, p2):
    if ENSEMBLE_CFG.get("kind") == "model" and ENSEMBLE_MODEL is not None:
        use_logits = ENSEMBLE_CFG.get("input", "probs") == "logits"
        X = np.stack([_to_logits(p1), _to_logits(p2)], axis=1) if use_logits else np.stack([p1, p2], axis=1)
        y = ENSEMBLE_MODEL.predict(X, verbose=0).squeeze()
        return np.atleast_1d(y).astype(float)
    # default: average
    return (0.5 * p1 + 0.5 * p2).astype(float)

def aggregate_clip(win_probs):
    return float(np.mean(win_probs) if AGGREGATE == "mean" else np.max(win_probs))

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
        "models_ready": (MODEL_1D is not None and MODEL_2D is not None),
        "ensemble_model_loaded": ENSEMBLE_MODEL is not None,
        "error": _LAST_LOAD_ERR,
        "window": WINDOW,
        "stride": STRIDE,
        "aggregate": AGGREGATE,
        "threshold": get_threshold(),
    })

@app.post("/predict")
def predict():
    # Only JSON mode here. Send { "x1d": [ [ [..4..]*T ]*N ] , "window_ids": ... }
    if not _safe_load_models():
        return jsonify({"error": "Model load failed", "detail": _LAST_LOAD_ERR}), 500

    payload = request.get_json(silent=True)
    if not payload or "x1d" not in payload:
        return jsonify({"error": "Send JSON with key 'x1d' shaped [N,T,4]"}), 400

    x1d_raw = np.array(payload["x1d"], dtype=np.float32)
    if x1d_raw.ndim != 3 or x1d_raw.shape[-1] != 4:
        return jsonify({"error": f"x1d must be [N,T,4], got {x1d_raw.shape}"}), 400

    # Build a basic 2D representation [N,T,4,1] by per-window min-max scale
    def to2d_batch(x1d):
        out = []
        for w in x1d:
            mn, mx = float(np.min(w)), float(np.max(w))
            normed = (w - mn) / (mx - mn + 1e-6)
            out.append(normed[..., None])
        return np.array(out, dtype=np.float32)

    x2d_raw = to2d_batch(x1d_raw)

    # Normalize by training stats
    x1d = normalize(x1d_raw, MU1D, SIG1D)
    x2d = normalize(x2d_raw, MU2D, SIG2D)

    try:
        p1 = np.atleast_1d(MODEL_1D.predict(x1d, verbose=0).squeeze())
        p2 = np.atleast_1d(MODEL_2D.predict(x2d, verbose=0).squeeze())
    except Exception as e:
        return jsonify({"error": "Inference failed", "detail": f"{type(e).__name__}: {e}"}), 500

    n = min(len(p1), len(p2))
    p1, p2 = p1[:n], p2[:n]
    win_probs = combine_window_probs(p1, p2)
    if len(win_probs) != n:
        win_probs = win_probs[:n]

    clip_prob = aggregate_clip(win_probs)
    th = get_threshold()
    decision = "Risky" if clip_prob >= th else "Safe"

    return jsonify({
        "decision": decision,
        "prob": float(clip_prob),
        "th": float(th),
        "window_probs": [float(v) for v in win_probs],
        "model_probs": {"p1d_mean": float(np.mean(p1)), "p2d_mean": float(np.mean(p2))},
        "meta": {"window": WINDOW, "stride": STRIDE, "aggregate": AGGREGATE}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)



