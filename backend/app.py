import os, json, tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import mixed_precision as mp
import cv2
import mediapipe as mpipe  # alias to avoid clashing with keras mixed_precision alias 'mp'

# ---- Patch legacy InputLayer configs (batch_shape -> batch_input_shape)
class PatchedInputLayer(K.layers.InputLayer):
    @classmethod
    def from_config(cls, config):
        if "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        return super().from_config(config)

# Also map DTypePolicy used in old graphs
CUSTOMS = {"InputLayer": PatchedInputLayer, "DTypePolicy": mp.Policy}

# Feature order from your CSV
FEATURE_NAMES = [
    "left_knee_angle",
    "right_knee_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
]

app = Flask(__name__)
CORS(app)

# ---------- small debug util ----------
def _shape(x):
    try:
        return list(x.shape)
    except Exception:
        return None

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

# Force 2D stats to broadcast as (1,1,4,1) against x2d [N,T,4,1]
def _as_2d_stat(arr_1d_or_2d):
    """
    Convert mu/sigma arrays like (4,), (4,1), (1,4) into shape (1,1,4,1)
    so broadcasting preserves the final channel dimension = 1.
    """
    a = np.array(arr_1d_or_2d, dtype=np.float32)
    if a.ndim == 1:            # (4,)
        a = a.reshape(1, 1, 4, 1)
    elif a.ndim == 2:          # (4,1) or (1,4)
        if a.shape == (4, 1):
            a = a.reshape(1, 1, 4, 1)
        elif a.shape == (1, 4):
            a = a.T.reshape(1, 1, 4, 1)
        else:
            a = a.reshape(-1)[:4].reshape(1, 1, 4, 1)
    else:
        a = a.squeeze()
        if a.ndim == 1 and a.size >= 4:
            a = a[:4].reshape(1, 1, 4, 1)
        else:
            a = np.zeros((1, 1, 4, 1), dtype=np.float32)
    return a

# ---------- Ensemble combine ----------
ENSEMBLE_MODEL = None  # optional
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

# ---------- Model builders (fallback when load_model fails) ----------
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

# ---------- 2D construction ----------
def to2d_batch_strict(x1d):
    """Return [N,T,4,1]. Collapse any accidental channels to 1."""
    out = []
    for w in x1d:                                # w: [T,4]
        mn, mx = float(np.min(w)), float(np.max(w))
        scale = (mx - mn) if (mx > mn) else 1.0
        normed = (w - mn) / (scale + 1e-6)       # [T,4]
        out.append(normed[..., None])            # [T,4,1]
    x2d = np.array(out, dtype=np.float32)        # [N,T,4,1]
    if x2d.ndim == 3:
        x2d = x2d[..., None]                     # ensure last dim exists
    if x2d.shape[-1] != 1:
        x2d = np.mean(x2d, axis=-1, keepdims=True)  # collapse -> 1 channel
    return x2d

# ---------- simple attribution on normalized 1D ----------
def _top_features_from_norm_window(z_win, top_k=2):
    """
    z_win: [T,4] absolute normalized values for one window.
    Returns top_k feature names with largest mean magnitude.
    """
    scores = z_win.mean(axis=0)  # (4,)
    order  = np.argsort(scores)[::-1]
    return [FEATURE_NAMES[i] for i in order[:top_k]]

def _overall_top_features(z_all, win_probs, th, top_k=2):
    """
    z_all: [N,T,4] abs normalized windows
    win_probs: [N]
    th: threshold for "risky" windows
    Returns overall top_k features aggregated over risky windows.
    """
    idx = np.where(win_probs >= th)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmax(win_probs))])
    # mean score per feature over selected windows
    feat_scores = np.mean([z_all[i].mean(axis=0) for i in idx], axis=0)  # (4,)
    order = np.argsort(feat_scores)[::-1]
    return [FEATURE_NAMES[i] for i in order[:top_k]]

# ---------- Shared predictor for JSON and video paths ----------
def predict_from_x1d(x1d_raw: np.ndarray):
    """x1d_raw: [N,T,4] in the CSV feature order. Returns Flask jsonify(...) + status."""
    # 1) Lazy load
    try:
        model_1d = get_model_1d()
        model_2d = get_model_2d()
    except Exception as e:
        return jsonify({"error": "Model load failed", "detail": f"{type(e).__name__}: {e}"}), 500

    if x1d_raw.ndim != 3 or x1d_raw.shape[-1] != 4:
        return jsonify({"error": f"x1d must be [N,T,4], got {list(x1d_raw.shape)}"}), 400

    # 2) Strict 2D and norms
    x2d_raw = to2d_batch_strict(x1d_raw)
    mu1, sg1 = _load_norm_file(os.path.join("norm", "cnn1d_data_anysafe.npz"))
    mu2, sg2 = _load_norm_file(os.path.join("norm", "cnn2d_data_anysafe.npz"))
    mu1, sg1 = _ensure_mu_sigma(mu1, sg1, feature_shape=(4,))
    mu2, sg2 = _ensure_mu_sigma(mu2, sg2, feature_shape=(4,))

    x1d = normalize(x1d_raw, mu1, sg1)
    mu2_e = _as_2d_stat(mu2); sg2_e = _as_2d_stat(sg2)
    x2d = (x2d_raw - mu2_e) / (sg2_e + 1e-6)

    # 3) Inference
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

    # 4) Simple attribution on normalized 1D
    z_all = np.abs(x1d[:n])                 # [N,T,4], abs normalized magnitudes
    per_window_top = []
    for i in range(n):
        per_window_top.append(_top_features_from_norm_window(z_all[i], top_k=2))
    risky_overall = _overall_top_features(z_all, win_probs, th, top_k=2)

    return jsonify({
        "decision": decision,
        "prob": float(clip_prob),
        "th": float(th),
        "window_probs": [float(v) for v in win_probs],
        "model_probs": {"p1d_mean": float(np.mean(p1)), "p2d_mean": float(np.mean(p2))},
        "risky_features_overall": risky_overall,
        "per_window_top_features": per_window_top,
        "meta": {"window": WINDOW, "stride": STRIDE, "aggregate": AGGREGATE}
    }), 200

# ---------- Video -> angles -> windows ----------
# MediaPipe landmark indices we need
_LS, _RS = 11, 12
_LE, _RE = 13, 14
_LH, _RH = 23, 24
_LK, _RK = 25, 26
_LA, _RA = 27, 28

def _angle_abc(a, b, c):
    """Angle at point b (degrees) for segments ba, bc. a,b,c are (x,y)."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def _frame_angles_landmarks(lm):
    """Return [left_knee, right_knee, left_shoulder, right_shoulder] from landmarks."""
    def pt(i): 
        return (lm[i].x, lm[i].y)
    # Knees: hip-knee-ankle
    lk = _angle_abc(pt(_LH), pt(_LK), pt(_LA))
    rk = _angle_abc(pt(_RH), pt(_RK), pt(_RA))
    # Shoulders: elbow-shoulder-hip
    ls = _angle_abc(pt(_LE), pt(_LS), pt(_LH))
    rs = _angle_abc(pt(_RE), pt(_RS), pt(_RH))
    return [lk, rk, ls, rs]

def extract_x1d_from_video(video_path: str, window: int = WINDOW, stride: int = STRIDE, max_frames: int = 5000):
    """Returns np.ndarray [N, window, 4] using MediaPipe Pose on the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    values = []  # per-frame [4]
    with mpipe.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or i >= max_frames: break
            i += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks is None:
                continue
            lm = res.pose_landmarks.landmark
            values.append(_frame_angles_landmarks(lm))
    cap.release()

    if len(values) == 0:
        raise RuntimeError("No pose detected in video")

    arr = np.array(values, dtype=np.float32)  # [F,4]

    # Make sliding windows [N, window, 4]
    F = arr.shape[0]
    if F < window:
        # pad by repeating last
        pad = np.tile(arr[-1], (window - F, 1))
        arr = np.vstack([arr, pad])
        F = arr.shape[0]

    windows = []
    for start in range(0, max(F - window + 1, 1), stride):
        end = start + window
        if end <= F:
            windows.append(arr[start:end])
    if not windows:
        windows.append(arr[-window:])

    x1d = np.stack(windows, axis=0).astype(np.float32)  # [N,window,4]
    return x1d

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

@app.post("/echo-shapes")
def echo_shapes():
    payload = request.get_json(silent=True) or {}
    x1d_raw = np.array(payload.get("x1d", []), dtype=np.float32)
    x2d_raw = to2d_batch_strict(x1d_raw)
    return jsonify({"x1d_raw": _shape(x1d_raw), "x2d_raw": _shape(x2d_raw)})

@app.post("/predict")
def predict():
    # 1) Read payload
    payload = request.get_json(silent=True)
    if not payload or "x1d" not in payload:
        return jsonify({"error": "Send JSON with key 'x1d' shaped [N,T,4]"}), 400

    x1d_raw = np.array(payload["x1d"], dtype=np.float32)
    return predict_from_x1d(x1d_raw)

@app.post("/analyze")
@app.post("/api/analyze")
def analyze_video():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded. Use multipart/form-data with field 'file'."}), 400

    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
        f.save(tmp.name)
        try:
            x1d = extract_x1d_from_video(tmp.name, window=WINDOW, stride=STRIDE)
        except Exception as e:
            return jsonify({"error": "Video parsing failed", "detail": f"{type(e).__name__}: {e}"}), 400

    return predict_from_x1d(x1d)

# ---- Warmup + single-worker debug (optional) ----
@app.get("/warmup")
def warmup():
    try:
        _ = get_model_1d()
        _ = get_model_2d()
        return jsonify({"ok": True, "models_ready": True, "pid": os.getpid()})
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)




