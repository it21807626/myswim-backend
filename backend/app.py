import os
import json
import tempfile
import math
import numpy as np
from numpy.linalg import norm

from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
from tensorflow import keras as K

# Keep everything strictly float32 (no mixed precision surprises)
K.mixed_precision.set_global_policy("float32")

# -------------------------------------------------------------------
# Shim: accept legacy 'batch_shape' from old .h5 into 'batch_input_shape'
# -------------------------------------------------------------------
class PatchedInputLayer(K.layers.InputLayer):
    @classmethod
    def from_config(cls, config):
        if "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        return super().from_config(config)

CUSTOMS = {
    "InputLayer": PatchedInputLayer
}

# -------------------------------------------------------------------
# App & CORS
# -------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------------------------------
# Manifest & global state
# -------------------------------------------------------------------
with open(os.path.join("config", "ensemble_manifest.json"), "r") as f:
    MANIFEST = json.load(f)

def _get_model_path(kind: str):
    for m in MANIFEST.get("models", []):
        if m.get("kind") == kind:
            return m.get("path")
    raise RuntimeError(f"No model path found for kind='{kind}' in manifest.")

M1D_PATH = _get_model_path("1d")     # e.g. models/cnn1d_anysafe_balanced.h5
M2D_PATH = _get_model_path("2d")     # e.g. models/cnn2d_anysafe_improved.h5

WINDOW  = int(MANIFEST.get("window", 30))
STRIDE  = int(MANIFEST.get("stride", 15))
ENSEMBLE_CFG = MANIFEST.get("ensemble", {})
AGGREGATE = ENSEMBLE_CFG.get("aggregate", MANIFEST.get("aggregate", "max")).lower()

# Globals loaded lazily
MODEL_1D = None
MODEL_2D = None
ENSEMBLE_MODEL = None
MU1D = SIG1D = MU2D = SIG2D = None
MODELS_READY = False
MODEL_LOAD_ERROR = None

# -------------------------------------------------------------------
# Lazy init loader
# -------------------------------------------------------------------
def init_models():
    """Load models and normalization files once, on first request."""
    global MODEL_1D, MODEL_2D, ENSEMBLE_MODEL
    global MU1D, SIG1D, MU2D, SIG2D, MODELS_READY, MODEL_LOAD_ERROR

    if MODELS_READY:
        return

    try:
        MODEL_1D = tf.keras.models.load_model(M1D_PATH, compile=False, custom_objects=CUSTOMS)
        MODEL_2D = tf.keras.models.load_model(M2D_PATH, compile=False, custom_objects=CUSTOMS)

        # Optional ensemble .h5
        ENSEMBLE_MODEL = None
        if ENSEMBLE_CFG.get("kind") == "model":
            ens_path = ENSEMBLE_CFG.get("path")
            if ens_path and os.path.exists(ens_path):
                ENSEMBLE_MODEL = tf.keras.models.load_model(ens_path, compile=False, custom_objects=CUSTOMS)

        # Norm stats
        n1 = np.load(os.path.join("norm", "cnn1d_data_anysafe.npz"))
        n2 = np.load(os.path.join("norm", "cnn2d_data_anysafe.npz"))
        MU1D, SIG1D = n1["mu"], n1["sigma"]
        MU2D, SIG2D = n2["mu"], n2["sigma"]

        MODELS_READY = True
        MODEL_LOAD_ERROR = None
    except Exception as e:
        MODELS_READY = False
        MODEL_LOAD_ERROR = f"{type(e).__name__}: {e}"

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def normalize(arr, mu, sigma, eps=1e-6):
    return (arr - mu) / (sigma + eps)

def _to_logits(p, eps=1e-7):
    p = np.clip(p, eps, 1. - eps)
    return np.log(p / (1. - p))

def combine_window_probs(p1, p2):
    """Combine 1D & 2D window-level probabilities according to manifest."""
    if ENSEMBLE_CFG.get("kind") == "model" and ENSEMBLE_MODEL is not None:
        in_kind = ENSEMBLE_CFG.get("input", "probs")
        if in_kind == "logits":
            X = np.stack([_to_logits(p1), _to_logits(p2)], axis=1)  # [N,2]
        else:
            X = np.stack([p1, p2], axis=1)
        y = ENSEMBLE_MODEL.predict(X, verbose=0).squeeze()
        return np.atleast_1d(y).astype(float)

    method = ENSEMBLE_CFG.get("method", "average")
    if method == "weighted":
        w = ENSEMBLE_CFG.get("weights", [0.5, 0.5])
        w1, w2 = float(w[0]), float(w[1])
        return (w1 * p1 + w2 * p2).astype(float)

    # default average
    return (0.5 * p1 + 0.5 * p2).astype(float)

def aggregate_clip(win_probs):
    if AGGREGATE == "mean":
        return float(np.mean(win_probs))
    return float(np.max(win_probs))

def get_threshold():
    th = ENSEMBLE_CFG.get("threshold", MANIFEST.get("threshold_risky", 0.463))
    return float(th)

# -------------------------------------------------------------------
# Pose → angles (MediaPipe)
# -------------------------------------------------------------------
import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose

def _angle(a, b, c):
    """Angle at b (degrees) between BA and BC."""
    ba, bc = a - b, c - b
    nba, nbc = norm(ba) + 1e-8, norm(bc) + 1e-8
    cosang = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

def _extract_angles_LK_RK_LS_RS(lm, img_w, img_h):
    """
    Feature order used in training:
      [left_knee, right_knee, left_shoulder, right_shoulder]
    MP indices:
      L: shoulder=11, elbow=13, hip=23, knee=25, ankle=27
      R: shoulder=12, elbow=14, hip=24, knee=26, ankle=28
    """
    def xy(i): return np.array([lm[i].x * img_w, lm[i].y * img_h], dtype=np.float32)
    LK = _angle(xy(23), xy(25), xy(27))
    RK = _angle(xy(24), xy(26), xy(28))
    LS = _angle(xy(13), xy(11), xy(23))
    RS = _angle(xy(14), xy(12), xy(24))
    return [LK, RK, LS, RS]

def _iter_frames(path, target_fps=15):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(src_fps / target_fps)), 1)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield frame
        i += 1
    cap.release()

def _video_to_feature_sequence(video_path, target_fps=15, angle_fn=_extract_angles_LK_RK_LS_RS):
    feats = []
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True
    ) as pose:
        for frame in _iter_frames(video_path, target_fps=target_fps):
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks is None:
                feats.append([np.nan, np.nan, np.nan, np.nan])
            else:
                lm = res.pose_landmarks.landmark
                feats.append(angle_fn(lm, w, h))
    feats = np.array(feats, dtype=np.float32)

    # fill NaNs
    if np.isnan(feats).any():
        for f in range(feats.shape[1]):
            col = feats[:, f]
            last = np.nan
            for i in range(len(col)):  # ffill
                if not np.isnan(col[i]):
                    last = col[i]
                else:
                    col[i] = last
            last = np.nan
            for i in range(len(col) - 1, -1, -1):  # bfill
                if not np.isnan(col[i]):
                    last = col[i]
                else:
                    col[i] = last if not np.isnan(last) else 0.0
        feats = feats.astype(np.float32)
    return feats  # [F_total, 4]

def _windowize(feats, T=30, stride=15):
    X, idx = [], []
    total = len(feats)
    for start in range(0, max(1, total - T + 1), stride):
        end = start + T
        if end <= total:
            X.append(feats[start:end])  # [T,4]
            idx.append((start, end))
    return np.array(X, dtype=np.float32), idx  # [N,T,4], list[(s,e)]

def _make_2d_from_1d_window(win_Tx4):
    # Min-max per window → [T,4,1]
    mn, mx = float(np.min(win_Tx4)), float(np.max(win_Tx4))
    normed = (win_Tx4 - mn) / (mx - mn + 1e-6)
    return normed[..., None].astype(np.float32)

def _build_2d_batch(x1d_batch):
    return np.stack([_make_2d_from_1d_window(w) for w in x1d_batch], axis=0)

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/")
def root():
    return jsonify({"message": "myswim backend up", "tf": tf.__version__})

@app.get("/health")
def health():
    status = {
        "ok": True,
        "models_ready": MODELS_READY,
        "error": MODEL_LOAD_ERROR,
        "window": WINDOW,
        "stride": STRIDE,
        "aggregate": AGGREGATE,
        "ensemble_model_loaded": ENSEMBLE_MODEL is not None if MODELS_READY else False,
        "threshold": get_threshold()
    }
    return jsonify(status)

@app.post("/predict")
def predict():
    # Ensure models are loaded
    if not MODELS_READY:
        init_models()
    if not MODELS_READY:
        # still failed → report cleanly
        return jsonify({"error": "models_not_ready", "detail": MODEL_LOAD_ERROR}), 503

    ct = (request.content_type or "").lower()

    # ---------- JSON mode ----------
    if "application/json" in ct:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Invalid JSON"}), 400

        if "x1d" not in payload:
            return jsonify({"error": "Missing 'x1d' (shape [N,T,4])"}), 400

        x1d_raw = np.array(payload["x1d"], dtype=np.float32)  # [N,T,4]
        if x1d_raw.ndim != 3 or x1d_raw.shape[-1] != 4:
            return jsonify({"error": f"x1d must be [N,T,4], got {x1d_raw.shape}"}), 400

        if "x2d" in payload:
            x2d_raw = np.array(payload["x2d"], dtype=np.float32)  # [N,T,4,1]
        else:
            x2d_raw = _build_2d_batch(x1d_raw)

        window_ids = payload.get("window_ids")

        # Normalize
        x1d = normalize(x1d_raw, MU1D, SIG1D)
        x2d = normalize(x2d_raw, MU2D, SIG2D)

        # Predict
        p1 = np.atleast_1d(MODEL_1D.predict(x1d, verbose=0).squeeze())
        p2 = np.atleast_1d(MODEL_2D.predict(x2d, verbose=0).squeeze())

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
            "window_ids": window_ids,
            "meta": {"window": WINDOW, "stride": STRIDE, "aggregate": AGGREGATE}
        })

    # ---------- Multipart (video) mode ----------
    if "multipart/form-data" in ct:
        if "video" not in request.files:
            return jsonify({"error": "No 'video' file part"}), 400

        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, "in.mp4")
            request.files["video"].save(fpath)

            # 1) frames → angles
            feats = _video_to_feature_sequence(fpath, target_fps=15)  # [F,4]

            # 2) windowize
            x1d_raw, win_idx = _windowize(feats, T=WINDOW, stride=STRIDE)  # [N,T,4]
            if len(x1d_raw) == 0:
                return jsonify({"error": "Video too short for one window"}), 400

            # 3) 2D build
            x2d_raw = _build_2d_batch(x1d_raw)  # [N,T,4,1]

            # 4) normalize
            x1d = normalize(x1d_raw, MU1D, SIG1D)
            x2d = normalize(x2d_raw, MU2D, SIG2D)

            # 5) predict
            p1 = np.atleast_1d(MODEL_1D.predict(x1d, verbose=0).squeeze())
            p2 = np.atleast_1d(MODEL_2D.predict(x2d, verbose=0).squeeze())
            n = min(len(p1), len(p2))
            p1, p2 = p1[:n], p2[:n]

            # 6) ensemble + aggregate
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
                "window_ids": win_idx,
                "meta": {"window": WINDOW, "stride": STRIDE, "aggregate": AGGREGATE}
            })

    return jsonify({"error": f"Unsupported Content-Type: {ct}"}), 415


if __name__ == "__main__":
    # Local dev; on Render we use gunicorn
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)


