import os, json, gc, math, tempfile
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import mixed_precision as mp
from flask import Flask, request, jsonify
from flask_cors import CORS

# Optional heavy deps; imported at top so cold start loads once
import cv2
import mediapipe as mp_solutions
from werkzeug.utils import secure_filename

# =======================
#   App / Runtime Limits
# =======================
app = Flask(__name__)
CORS(app)

# Reject very large uploads (change via env if needed)
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH_MB', '40')) * 1024 * 1024  # 40 MB

# Lightweight decode knobs (override via env)
TARGET_WIDTH  = int(os.environ.get('TARGET_WIDTH',  '480'))   # resize frames to <= this width
TARGET_FPS    = int(os.environ.get('TARGET_FPS',    '10'))    # sample to ~N fps
MAX_SECONDS   = int(os.environ.get('MAX_SECONDS',   '15'))    # cap per request

# ================
#  Model Plumbing
# ================
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
    if a.ndim == 1:
        a = a.reshape(1, 1, 4, 1)
    elif a.ndim == 2:
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
    """Try normal load_model (with CUSTOMS). If legacy-graph errors occur,
    rebuild the architecture and load only weights by name."""
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
        x2d = x2d[..., None]
    if x2d.shape[-1] != 1:
        x2d = np.mean(x2d, axis=-1, keepdims=True)
    return x2d

# ---------- simple attribution on normalized 1D ----------
def _top_features_from_norm_window(z_win, top_k=2):
    """z_win: [T,4] absolute normalized values for one window -> top_k names."""
    scores = z_win.mean(axis=0)  # (4,)
    order  = np.argsort(scores)[::-1]
    return [FEATURE_NAMES[i] for i in order[:top_k]]

def _overall_top_features(z_all, win_probs, th, top_k=2):
    """Aggregate over risky windows; fallback to the max-prob window."""
    idx = np.where(win_probs >= th)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmax(win_probs))])
    feat_scores = np.mean([z_all[i].mean(axis=0) for i in idx], axis=0)  # (4,)
    order = np.argsort(feat_scores)[::-1]
    return [FEATURE_NAMES[i] for i in order[:top_k]]

# -----------------------------
#    Pose → 4 angles helpers
# -----------------------------
LM = mp_solutions.solutions.pose.PoseLandmark

def _angle(a, b, c) -> float:
    """Return angle at point b (in degrees) for points a-b-c, given as (x,y)."""
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = np.array([ax - bx, ay - by], dtype=np.float32)
    v2 = np.array([cx - bx, cy - by], dtype=np.float32)
    dot = float(np.dot(v1, v2))
    n1 = float(np.linalg.norm(v1)) + 1e-8
    n2 = float(np.linalg.norm(v2)) + 1e-8
    cosang = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def _extract_frame_angles(lms) -> Tuple[float, float, float, float]:
    """Return (L_knee, R_knee, L_shoulder, R_shoulder) degrees."""
    # Coordinates in image-space (x,y)
    def xy(idx):
        p = lms.landmark[idx]
        return (p.x, p.y)

    # Knees: hip-knee-ankle
    lk = _angle(xy(LM.LEFT_HIP),  xy(LM.LEFT_KNEE),  xy(LM.LEFT_ANKLE))
    rk = _angle(xy(LM.RIGHT_HIP), xy(LM.RIGHT_KNEE), xy(LM.RIGHT_ANKLE))

    # Shoulders: elbow-shoulder-hip
    ls = _angle(xy(LM.LEFT_ELBOW),  xy(LM.LEFT_SHOULDER),  xy(LM.LEFT_HIP))
    rs = _angle(xy(LM.RIGHT_ELBOW), xy(LM.RIGHT_SHOULDER), xy(LM.RIGHT_HIP))

    return (lk, rk, ls, rs)

def _video_to_feature_sequence(video_path: str) -> Tuple[np.ndarray, float]:
    """Decode video with downscale/FPS sampling/cap. Return features [F,4] and approx fps_used."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_every = max(1, int(round(fps / max(1, TARGET_FPS))))
    max_frames = int(MAX_SECONDS * fps)
    processed = 0
    feats: List[Tuple[float, float, float, float]] = []

    with mp_solutions.solutions.pose.Pose(
        static_image_mode=False, model_complexity=0,
        enable_segmentation=False, smooth_landmarks=True
    ) as pose:
        while True:
            # Skip frames to hit target FPS
            for _ in range(sample_every - 1):
                if not cap.grab():
                    break
                processed += 1
                if processed >= max_frames:
                    break

            ok, frame = cap.read()
            if not ok or processed >= max_frames:
                break
            processed += 1

            # Downscale to limit RAM/CPU
            if TARGET_WIDTH and frame.shape[1] > TARGET_WIDTH:
                scale = TARGET_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, (int(frame.shape[1] * scale),
                                           int(frame.shape[0] * scale)))

            # BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            lk, rk, ls, rs = _extract_frame_angles(res.pose_landmarks)
            feats.append((lk, rk, ls, rs))

    cap.release()
    gc.collect()
    if len(feats) == 0:
        raise RuntimeError("No pose landmarks detected")

    fps_used = max(1.0, fps / sample_every)
    return np.array(feats, dtype=np.float32), float(fps_used)

def _windows_from_sequence(seq_f4: np.ndarray, win: int, stride: int) -> np.ndarray:
    """seq_f4: [F,4] → [N,win,4] sliding windows; pad tail if too short."""
    F = seq_f4.shape[0]
    if F < win:
        pad = np.repeat(seq_f4[-1:], win - F, axis=0)
        return np.expand_dims(np.concatenate([seq_f4, pad], axis=0), 0)

    windows = []
    for s in range(0, F - win + 1, stride):
        windows.append(seq_f4[s:s+win])
    if not windows:
        windows.append(seq_f4[-win:])
    return np.stack(windows, axis=0)

# -----------------------------
#   Quick "is swimming?" gate
# -----------------------------
def _blue_ratio(frame_bgr):
    """Return fraction of 'blue-ish' pixels in frame (0..1)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # OpenCV H range: 0..179; pool water often ~90..140, widen a bit:
    lower = np.array([85,  40,  40], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask)) / float(mask.size)

def quick_swim_check(video_path: str,
                     sample_every:int = 5,     # sample every Nth frame
                     max_samples:int = 150,    # cap work
                     min_pose_frames:int = 12, # >= this many frames with pose
                     min_avg_blue:float = 0.10 # >=10% blue-ish pixels on average
                     ):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for validation")

    pose_frames = 0
    blue_scores = []

    with mp_solutions.solutions.pose.Pose(
        static_image_mode=False, model_complexity=0,
        enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        i = 0
        taken = 0
        while taken < max_samples:
            ok = cap.grab()
            if not ok:
                break
            if i % sample_every == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                blue_scores.append(_blue_ratio(frame))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if res.pose_landmarks is not None:
                    pose_frames += 1
                taken += 1
            i += 1

    cap.release()

    if not blue_scores:
        return {"is_swimming": False, "pose_frames": 0, "avg_blue": 0.0}

    avg_blue = float(np.mean(blue_scores))
    is_swim = (pose_frames >= min_pose_frames) and (avg_blue >= min_avg_blue)
    return {"is_swimming": bool(is_swim), "pose_frames": int(pose_frames), "avg_blue": avg_blue}


# ------------
#  Inference
# ------------
def _infer_on_x1d(x1d_raw: np.ndarray):
    """Run both models on x1d_raw [N,T,4] and return the final JSON payload."""
    # 1) Models
    try:
        model_1d = get_model_1d()
        model_2d = get_model_2d()
    except Exception as e:
        return {"error": "Model load failed", "detail": f"{type(e).__name__}: {e}"}, 500

    # 2) Build strict 2D [N,T,4,1]
    x2d_raw = to2d_batch_strict(x1d_raw)

    # 3) Norms
    mu1, sg1 = _load_norm_file(os.path.join("norm", "cnn1d_data_anysafe.npz"))
    mu2, sg2 = _load_norm_file(os.path.join("norm", "cnn2d_data_anysafe.npz"))
    mu1, sg1 = _ensure_mu_sigma(mu1, sg1, feature_shape=(4,))
    mu2, sg2 = _ensure_mu_sigma(mu2, sg2, feature_shape=(4,))

    x1d = normalize(x1d_raw, mu1, sg1)
    mu2_e = _as_2d_stat(mu2)     # (1,1,4,1)
    sg2_e = _as_2d_stat(sg2)     # (1,1,4,1)
    x2d = (x2d_raw - mu2_e) / (sg2_e + 1e-6)

    # 4) Inference
    try:
        p1 = np.atleast_1d(model_1d.predict(x1d, verbose=0).squeeze())
        p2 = np.atleast_1d(model_2d.predict(x2d, verbose=0).squeeze())
    except Exception as e:
        return {"error": "Inference failed", "detail": f"{type(e).__name__}: {e}"}, 500

    n = min(len(p1), len(p2))
    p1, p2 = p1[:n], p2[:n]
    win_probs = combine_window_probs(p1, p2)
    if len(win_probs) != n:
        win_probs = win_probs[:n]

    clip_prob = aggregate_clip(win_probs)
    th = get_threshold()
    decision = "Risky" if clip_prob >= th else "Safe"

    # 5) Simple per-feature attribution on normalized 1D
    z_all = np.abs(x1d[:n])                 # [N,T,4], abs normalized magnitudes
    per_window_top = []
    for i in range(n):
        per_window_top.append(_top_features_from_norm_window(z_all[i], top_k=2))
    risky_overall = _overall_top_features(z_all, win_probs, th, top_k=2)

    payload = {
        "decision": decision,
        "prob": float(clip_prob),
        "th": float(th),
        "window_probs": [float(v) for v in win_probs],
        "model_probs": {"p1d_mean": float(np.mean(p1)), "p2d_mean": float(np.mean(p2))},
        "risky_features_overall": risky_overall,
        "per_window_top_features": per_window_top,
        "meta": {"window": WINDOW, "stride": STRIDE, "aggregate": AGGREGATE}
    }
    return payload, 200

# ===========
#   Routes
# ===========
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
        "limits": {
            "TARGET_WIDTH": TARGET_WIDTH,
            "TARGET_FPS": TARGET_FPS,
            "MAX_SECONDS": MAX_SECONDS,
            "MAX_CONTENT_MB": app.config['MAX_CONTENT_LENGTH'] // (1024*1024),
        }
    })

@app.post("/echo-shapes")
def echo_shapes():
    payload = request.get_json(silent=True) or {}
    x1d_raw = np.array(payload.get("x1d", []), dtype=np.float32)
    x2d_raw = to2d_batch_strict(x1d_raw)
    return jsonify({"x1d_raw": _shape(x1d_raw), "x2d_raw": _shape(x2d_raw)})

@app.post("/predict")
def predict():
    """Raw JSON input: {"x1d": [[...],[...],...]} shaped [N,T,4]."""
    payload = request.get_json(silent=True)
    if not payload or "x1d" not in payload:
        return jsonify({"error": "Send JSON with key 'x1d' shaped [N,T,4]"}), 400

    x1d_raw = np.array(payload["x1d"], dtype=np.float32)
    if x1d_raw.ndim != 3 or x1d_raw.shape[-1] != 4:
        return jsonify({"error": f"x1d must be [N,T,4], got {x1d_raw.shape}"}), 400

    # Optional: shapes only
    if request.args.get("shapes") == "1":
        return jsonify({
            "x1d_raw": _shape(x1d_raw),
            "x2d_raw": _shape(to2d_batch_strict(x1d_raw)),
        })

    out, code = _infer_on_x1d(x1d_raw)
    return (jsonify(out), code)

@app.post("/validate")
@app.post("/api/validate")
def validate_video():
    if 'file' not in request.files:
        return jsonify({"error": "Upload a file under form field 'file'"}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    fname = secure_filename(f.filename)
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = os.path.join(td, fname or "clip.mp4")
            f.save(tmp_path)
            res = quick_swim_check(tmp_path)
            res = quick_swim_check(tmp_path)
            return jsonify({"ok": True, **res}), 200


    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 400


@app.post("/api/analyze")
def analyze_video():
    """Accepts multipart/form-data with 'file' (mp4/mov), extracts 4 angles,
       builds [N,T,4] windows, runs inference, and returns the same JSON schema."""
    if 'file' not in request.files:
        return jsonify({"error": "Upload a file under form field 'file'"}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    fname = secure_filename(f.filename)
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = os.path.join(td, fname or "clip.mp4")
            f.save(tmp_path)

            gate = quick_swim_check(tmp_path)
            if not gate.get("is_swimming", False):
                return jsonify({
                    "error": "Not a swimming clip",
                    "is_swimming": False,
                    "pose_frames": gate.get("pose_frames", 0),
                    "avg_blue": gate.get("avg_blue", 0.0)
                }), 422


            # Video → per-frame 4 angles
            seq_f4, fps_used = _video_to_feature_sequence(tmp_path)

            # Sequence → [N,WINDOW,4]
            x1d_raw = _windows_from_sequence(seq_f4, WINDOW, STRIDE)

            out, code = _infer_on_x1d(x1d_raw)
            if code != 200:
                return jsonify(out), code

            # include fps used for optional UI time-axis
            out["meta"]["fps_used"] = fps_used
            return jsonify(out), 200
    except Exception as e:
        return jsonify({"error": "Video analysis failed", "detail": f"{type(e).__name__}: {e}"}), 500
    finally:
        gc.collect()

@app.get("/warmup")
def warmup():
    try:
        _ = get_model_1d()
        _ = get_model_2d()
        return jsonify({"ok": True, "models_ready": True, "pid": os.getpid()})
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500

if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
