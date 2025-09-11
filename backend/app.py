import os, json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import mixed_precision as mp

# ---- Patch legacy InputLayer configs (batch_shape -> batch_input_shape)
class PatchedInputLayer(K.layers.InputLayer):
    @classmethod
    def from_config(cls, config):
        if "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        return super().from_config(config)

# Also map DTypePolicy used in old graphs
CUSTOMS = {"InputLayer": PatchedInputLayer, "DTypePolicy": mp.Policy}

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

# ---------- Robust norm helpers ----------
def _pick_key(npz, *candidates):
    for k in candidates:
        if k in npz.files:
            return k
    return None

def _load_norm_file(path):
    mu = None
    sigma = None
    try:
        npz = np.load(path)
        k_mu = _pick_key(npz, "mu", "mean", "avg")
        k_sg = _pick_key(npz, "sigma", "std", "stddev", "stdev")
        if k_mu is not None: mu = np.array(npz[k_mu], dtype=np.float32)
        if k_sg is not None: sigma = np.array(npz[k_sg], dtype=np.float32)
    except Exception:
        pass
    return mu, sigma

def _ensure_mu_sigma(mu, sigma, feature_shape):
    if mu is None:    mu = np.zeros(feature_shape, dtype=np.float32)
    if sigma is None: sigma = np.ones(feature_shape, dtype=np.float32)
    return mu, sigma

def normalize(arr, mu, sigma, eps=1e-6):
    return (arr - mu) / (sigma + eps)

def _to_logits(p, eps=1e-7):
    p = np.clip(p, eps, 1. - eps)
    return np.log(p/(1.-p))

# ---------- Ensemble combine ----------
ENSEMBLE_MODEL = None  # (optional) if you later want to lazy-load an ensemble model
def combine_window_probs(p1, p2):
    if ENSEMBLE_CFG.get("kind") == "model" and ENSEMBLE_MODEL is not None:
        use_logits = ENSEMBLE_CFG.get("input", "probs") == "logits"
        X = np.stack([_to_logits(p1), _to_logits(p2)], axis=1) if use_logits else np.stack([p1, p2], axis=1)
        y = ENSEMBLE_MODEL.predict(X, verbose=0).squeeze()
        return np.atleast_1d(y).astype(float)
    return (0.5 * p1 + 0.5 * p2).astype(float)

def aggregate_clip(win_probs):
    return float(np.mean(win_probs) if AGGREGATE == "mean" else np.max(win_probs))

def get_threshold():
    th = ENSEMBLE_CFG.get("threshold", MANIFEST.get("threshold_risky", 0.463))
    return float(th)

# ---------- Model builders (match your old layer names/shapes) ----------
def build_cnn1d(input_shape=(30, 4)):
    x_in = K.layers.Input(shape=input_shape, name="input_layer")
    x = K.layers.Conv1D(64, 7, padding="same", activation="relu", name="conv1d")(x_in)
    x = K.layers.BatchNormalization(name="batch_normalization")(x)
    x = K.layers.MaxPooling1D(pool_size=2, strides=2, name="max_pooling1d")(x)
    x = K.layers.Conv1D(128, 5, padding="same", activation="relu", name="conv1d_1")(x)
    x = K.layers.BatchNormalization(name="batch_normalization_1")(x)
    x = K.layers.MaxPooling1D(pool_size=2, strides=2, name="max_pooling1d_1")(x)
    x = K.layers.Conv1D(256, 3, padding="same", activation="relu", name="conv1d_2")(x)
    x = K.layers.GlobalAveragePooling1D(name="global_average_pooling1d")(x)
    x = K.layers.Dropout(0.3, name="dropout")(x)
    x = K.layers.Dense(128, activation="relu", name="dense")(x)
    out = K.layers.Dense(1, activation="sigmoid", name="dense_1")(x)
    return K.Model(x_in, out, name="functional")

def build_cnn2d(input_shape=(30, 4, 1)):
    x_in = K.layers.Input(shape=input_shape, name="input_layer_3")
    x = K.layers.Conv2D(32, (5, 2), padding="same", activation="relu", name="conv2d_18")(x_in)
    x = K.layers.BatchNormalization(name="batch_normalization_15")(x)
    x = K.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name="max_pooling2d_12")(x)
    x = K.layers.Conv2D(64, (3, 2), padding="same", activation="relu", name="conv2d_19")(x)
    x = K.layers.BatchNormalization(name="batch_normalization_16")(x)
    x = K.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name="max_pooling2d_13")(x)
    x = K.layers.Conv2D(128, (3, 1), padding="same", activation="relu", name="conv2d_20")(x)
    x = K.layers.BatchNormalization(name="batch_normalization_17")(x)
    x = K.layers.GlobalAveragePooling2D(name="global_average_pooling2d_3")(x)
    x = K.layers.Dropout(0.35, name="dropout_3")(x)
    x = K.layers.Dense(128, activation="relu", name="dense_6")(x)
    out = K.layers.Dense(1, activation="sigmoid", name="dense_7")(x)
    return K.Model(x_in, out, name="functional_3")

def robust_load_or_rebuild(h5_path: str, kind: str):
    """
    Try normal load_model (with CUSTOMS). If legacy-graph errors occur,
    rebuild the architecture and load only weights by name.
    """
    try:
        return tf.keras.models.load_model(h5_path, compile=False, custom_objects=CUSTOMS)
    except Exception:
        if kind == "1d":
            model = build_cnn1d()
        elif kind == "2d":
            model = build_cnn2d()
        else:
            raise RuntimeError(f"Unknown kind: {kind}")
        # First try strict by_name load; if that fails, allow skip_mismatch
        try:
            model.load_weights(h5_path, by_name=True, skip_mismatch=False)
        except Exception:
            model.load_weights(h5_path, by_name=True, skip_mismatch=True)
        return model

# ---------- Lazy model singletons ----------
_MODEL_1D = None
_MODEL_2D = None
_LAST_LOAD_ERR = None

def get_model_1d():
    global _MODEL_1D, _LAST_LOAD_ERR
    if _MODEL_1D is None:
        try:
            _MODEL_1D = robust_load_or_rebuild(M1D_PATH, "1d")
            _LAST_LOAD_ERR = None
        except Exception as e:
            _LAST_LOAD_ERR = f"{type(e).__name__}: {e}"
            raise
    return _MODEL_1D

def get_model_2d():
    global _MODEL_2D, _LAST_LOAD_ERR
    if _MODEL_2D is None:
        try:
            _MODEL_2D = robust_load_or_rebuild(M2D_PATH, "2d")
            _LAST_LOAD_ERR = None
        except Exception as e:
            _LAST_LOAD_ERR = f"{type(e).__name__}: {e}"
            raise
    return _MODEL_2D

# ---------- Routes ----------
@app.get("/")
def root():
    return jsonify({"message": "myswim backend up", "tf": tf.__version__})

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "models_ready": (_MODEL_1D is not None and _MODEL_2D is not None),
        "ensemble_model_loaded": ENSEMBLE_MODEL is not None,
        "error": _LAST_LOAD_ERR,
        "window": WINDOW,
        "stride": STRIDE,
        "aggregate": AGGREGATE,
        "threshold": get_threshold(),
    })

@app.post("/predict")
def predict():
    # 1) Lazy load models
    try:
        model_1d = get_model_1d()
        model_2d = get_model_2d()
    except Exception as e:
        return jsonify({"error": "Model load failed", "detail": f"{type(e).__name__}: {e}"}), 500

    # 2) Read payload
    payload = request.get_json(silent=True)
    if not payload or "x1d" not in payload:
        return jsonify({"error": "Send JSON with key 'x1d' shaped [N,T,4]"}), 400

    x1d_raw = np.array(payload["x1d"], dtype=np.float32)
    if x1d_raw.ndim != 3 or x1d_raw.shape[-1] != 4:
        return jsonify({"error": f"x1d must be [N,T,4], got {x1d_raw.shape}"}), 400

    # 3) Build simple 2D representation [N,T,4,1]
    def to2d_batch(x1d):
        out = []
        for w in x1d:
            mn, mx = float(np.min(w)), float(np.max(w))
            normed = (w - mn) / (mx - mn + 1e-6)
            out.append(normed[..., None])
        return np.array(out, dtype=np.float32)

    x2d_raw = to2d_batch(x1d_raw)

    # 4) Load/ensure norms
    mu1, sg1 = _load_norm_file(os.path.join("norm", "cnn1d_data_anysafe.npz"))
    mu2, sg2 = _load_norm_file(os.path.join("norm", "cnn2d_data_anysafe.npz"))
    mu1, sg1 = _ensure_mu_sigma(mu1, sg1, feature_shape=(4,))
    mu2, sg2 = _ensure_mu_sigma(mu2, sg2, feature_shape=(4,))

    x1d = normalize(x1d_raw, mu1, sg1)
    x2d = normalize(
        x2d_raw,
        mu2[..., None] if mu2.ndim == 1 else mu2,
        sg2[..., None] if sg2.ndim == 1 else sg2
    )

    # 5) Inference
    try:
        p1 = np.atleast_1d(model_1d.predict(x1d, verbose=0).squeeze())
        p2 = np.atleast_1d(model_2d.predict(x2d, verbose=0).squeeze())
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




