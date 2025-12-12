# camera.py
"""
Threaded webcam reader + emotion predictor (robust + debug-friendly).

Place:
 - your .h5 Keras/TensorFlow model (model_best.h5 or model_resumed_final.h5)
 - haarcascade_frontalface_default.xml
 - class_indices.json (optional, produced by training script)
in the same folder as this file.

Run: python camera.py   (press 'q' to quit the preview window)
"""

import os
import time
import threading
import traceback
from threading import Thread
from collections import deque

import cv2
import numpy as np
import pandas as pd
import json
import sys

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
)

# -------------------------------
# Config / paths (edit if needed)
# -------------------------------
CASCADE_PATH = "haarcascade_frontalface_default.xml"
MODEL_PATH = "model_final.h5"   # change to model_best.h5 if you have that
CLASS_MAP_JSON = "class_indices.json"
DEFAULT_WINDOW_LEN = 5
DEFAULT_CONF_THRESHOLD = 0.40   # set ~0.55 for production; lower for debug
DEBUG_PRED = True               # set True to print prediction arrays and diagnostics

# fallback mapping if class_indices can't be read
fallback_emotion_dict = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"
}

music_dist = {
    0: "songs/angry.csv",
    1: "songs/disgusted.csv",
    2: "songs/fearful.csv",
    3: "songs/happy.csv",
    4: "songs/neutral.csv",
    5: "songs/sad.csv",
    6: "songs/surprised.csv"
}

cv2.ocl.setUseOpenCL(False)

# -------------------------------
# Model architecture helper
# -------------------------------
def build_emotion_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model

# -------------------------------
# Robust model loader
# -------------------------------
def load_emotion_model(model_path):
    if not os.path.exists(model_path):
        print(f"[camera.py] MODEL_PATH '{model_path}' not found.")
        return None
    model = None
    # Try tf.keras.load_model first
    try:
        model = load_model(model_path, compile=False)
        print(f"[camera.py] Loaded full model from '{model_path}' (compile=False).")
    except Exception as e:
        print(f"[camera.py] load_model raised: {e}\nTrying fallback: load weights into same-arch model...")
        traceback.print_exc()
        try:
            model = build_emotion_model()
            model.load_weights(model_path)
            print(f"[camera.py] Loaded weights into built architecture from '{model_path}'.")
        except Exception as e2:
            print(f"[camera.py] Failed to load weights fallback: {e2}")
            traceback.print_exc()
            model = None
    # Optional small compile so predict() is silent about optimizer state
    if model is not None:
        try:
            model.compile(optimizer='adam', loss='categorical_crossentropy')
        except Exception:
            pass
    return model

# -------------------------------
# Load model + class mapping
# -------------------------------
emotion_model = load_emotion_model(MODEL_PATH)

# Build emotion_dict robustly from class_indices.json (support both index->label and label->index)
emotion_dict = fallback_emotion_dict.copy()
if os.path.exists(CLASS_MAP_JSON):
    try:
        with open(CLASS_MAP_JSON, "r") as fh:
            class_map = json.load(fh)
        # Detect format
        # If keys are numeric strings -> index->label mapping (e.g. {"0":"Angry",...})
        keys = list(class_map.keys())
        if len(keys) > 0 and all(k.strip().isdigit() for k in keys):
            # index->label
            inv = {int(k): v for k, v in class_map.items()}
            if DEBUG_PRED:
                print(f"[camera.py] Detected class_map format: index->label. Parsed: {inv}")
            for i in range(7):
                if i in inv:
                    emotion_dict[i] = inv[i]
        else:
            # assume label->index (e.g. {"Angry":0,...}), invert
            inv = {}
            for label, idx in class_map.items():
                try:
                    inv[int(idx)] = label
                except Exception:
                    # if idx not int, store as-is
                    inv[idx] = label
            if DEBUG_PRED:
                print(f"[camera.py] Detected class_map format: label->index or mixed. Parsed inv: {inv}")
            for i in range(7):
                if i in inv:
                    emotion_dict[i] = inv[i]
    except Exception as e:
        print(f"[camera.py] Could not parse {CLASS_MAP_JSON}: {e}")
        traceback.print_exc()
else:
    if DEBUG_PRED:
        print(f"[camera.py] {CLASS_MAP_JSON} not found; using fallback mapping.")

if DEBUG_PRED:
    print("[camera.py] Using emotion_dict:", emotion_dict)

# -------------------------------
# Load Haar cascade
# -------------------------------
face_cascade = None
if os.path.exists(CASCADE_PATH):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print(f"[camera.py] Cascade loaded but empty/invalid: {CASCADE_PATH}")
        face_cascade = None
    else:
        print(f"[camera.py] Loaded Haar cascade from '{CASCADE_PATH}'")
else:
    print(f"[camera.py] Cascade file '{CASCADE_PATH}' not found. Face detection will fail without it.")

# -------------------------------
# Prediction helper
# -------------------------------
def _predict_emotion_from_model(face_gray, model, debug=False):
    if model is None:
        return None, 0.0
    try:
        if face_gray is None or face_gray.size == 0:
            if debug:
                print("[camera.py] Empty face crop; skipping predict.")
            return None, 0.0
        # ensure shape is correct
        face_resized = cv2.resize(face_gray, (48, 48))
        arr = face_resized.astype('float32') / 255.0
        arr = np.expand_dims(arr, axis=0)   # (1,48,48)
        arr = np.expand_dims(arr, axis=-1)  # (1,48,48,1)
        preds = model.predict(arr, verbose=0)
        preds = np.asarray(preds).reshape(-1)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        if debug or DEBUG_PRED:
            print(f"[camera.py] predict -> idx={idx}, conf={conf:.4f}, preds={np.round(preds,4)}")
        return idx, conf
    except Exception as e:
        if DEBUG_PRED:
            print("[camera.py] Prediction error:", e)
            traceback.print_exc()
        return None, 0.0

# -------------------------------
# Threaded capture class
# -------------------------------
class WebcamVideoStream:
    def __init__(self, src=0):
        self.src = src
        try:
            self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.stream.isOpened():
                self.stream = cv2.VideoCapture(src)
        except Exception:
            self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.thread is None:
            self.thread = Thread(target=self.update, daemon=True)
            self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            try:
                grabbed, frame = self.stream.read()
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except Exception:
                pass
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        try:
            if hasattr(self.stream, "release"):
                self.stream.release()
        except Exception:
            pass

# -------------------------------
# VideoCamera wrapper
# -------------------------------
class VideoCamera(object):
    def __init__(self, src=0, window_len=DEFAULT_WINDOW_LEN, conf_threshold=DEFAULT_CONF_THRESHOLD):
        self.src = src
        self.cap = WebcamVideoStream(src=src).start()
        self.window_len = int(window_len)
        self.conf_threshold = float(conf_threshold)
        self.emotion_window = deque(maxlen=self.window_len)
        self.current_emotion = 4
        self.lock = threading.Lock()

    def stop(self):
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.stop()
        except Exception:
            pass

    def _predict_emotion(self, face_gray):
        return _predict_emotion_from_model(face_gray, emotion_model, debug=False)

    def _read_recs_for_emotion(self, emotion_index, top_n=15):
        try:
            csv_path = music_dist.get(int(emotion_index))
            if csv_path is None or not os.path.exists(csv_path):
                return pd.DataFrame(columns=["Name", "Album", "Artist"])
            df = pd.read_csv(csv_path)
            cols = df.columns.tolist()
            rename_map = {}
            if "Name" not in cols and len(cols) > 0:
                rename_map[cols[0]] = "Name"
            if "Album" not in cols and len(cols) > 1:
                rename_map[cols[1]] = "Album"
            if "Artist" not in cols and len(cols) > 2:
                rename_map[cols[2]] = "Artist"
            if rename_map:
                df = df.rename(columns=rename_map)
            df = df.reindex(columns=["Name", "Album", "Artist"])
            return df.head(top_n)
        except Exception:
            return pd.DataFrame(columns=["Name", "Album", "Artist"])

    def get_frame(self):
        frame = self.cap.read()
        if frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, jpeg = cv2.imencode('.jpg', blank)
            return jpeg.tobytes(), pd.DataFrame(columns=["Name", "Album", "Artist"])

        disp = cv2.resize(frame, (640, 480))
        df_local = self._read_recs_for_emotion(self.current_emotion)

        if face_cascade is not None:
            gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            if len(faces) == 0:
                with self.lock:
                    self.emotion_window.clear()
            else:
                faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
                x, y, w, h = faces[0]
                # ensure non-empty ROI
                if w <= 0 or h <= 0:
                    if DEBUG_PRED:
                        print("[camera.py] Detected face with zero area, skipping.")
                else:
                    cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face_gray = gray[y:y + h, x:x + w]
                    pred_idx, conf = self._predict_emotion(face_gray)
                    if pred_idx is not None:
                        # append only if strong enough (but still allow showing last pred)
                        if conf >= self.conf_threshold:
                            with self.lock:
                                self.emotion_window.append(pred_idx)
                        with self.lock:
                            try:
                                final_idx = max(set(self.emotion_window), key=self.emotion_window.count) if len(self.emotion_window) > 0 else pred_idx
                            except Exception:
                                final_idx = pred_idx
                            self.current_emotion = int(final_idx)
                        label = emotion_dict.get(self.current_emotion, str(self.current_emotion))
                        cv2.putText(disp, f"{label}", (x + 10, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        df_local = self._read_recs_for_emotion(self.current_emotion)

        try:
            ret, jpeg = cv2.imencode('.jpg', disp)
            jpeg_bytes = jpeg.tobytes()
        except Exception:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, jpeg = cv2.imencode('.jpg', blank)
            jpeg_bytes = jpeg.tobytes()

        return jpeg_bytes, df_local

    def capture_once(self):
        try:
            frame = self.cap.read()
        except Exception:
            frame = None

        if frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, jpeg = cv2.imencode('.jpg', blank)
            return jpeg.tobytes(), pd.DataFrame(columns=["Name", "Album", "Artist"])

        disp = cv2.resize(frame, (640, 480))
        df_local = self._read_recs_for_emotion(self.current_emotion)

        if face_cascade is not None:
            gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            if len(faces) > 0:
                faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
                x, y, w, h = faces[0]
                if w > 0 and h > 0:
                    cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face_gray = gray[y:y + h, x:x + w]
                    pred_idx, conf = self._predict_emotion(face_gray)
                    if pred_idx is not None:
                        chosen_idx = int(pred_idx)
                        label = emotion_dict.get(chosen_idx, str(chosen_idx))
                        cv2.putText(disp, f"{label} ({conf:.2f})", (x + 10, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        df_local = self._read_recs_for_emotion(chosen_idx)

        try:
            ret, jpeg = cv2.imencode('.jpg', disp)
            jpeg_bytes = jpeg.tobytes()
        except Exception:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, jpeg = cv2.imencode('.jpg', blank)
            jpeg_bytes = jpeg.tobytes()

        return jpeg_bytes, df_local

# simple fallback music rec
def music_rec():
    try:
        df = pd.read_csv(music_dist.get(4))
        df = df.reindex(columns=["Name", "Album", "Artist"]).head(15)
        return df
    except Exception:
        return pd.DataFrame(columns=["Name", "Album", "Artist"])

# -------------------------------
# Test harness when run directly
# -------------------------------
if __name__ == "__main__":
    cam = VideoCamera(src=0, window_len=DEFAULT_WINDOW_LEN, conf_threshold=DEFAULT_CONF_THRESHOLD)
    print("Starting webcam preview. Press 'q' to quit.")
    try:
        while True:
            jpeg_bytes, df = cam.get_frame()
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                print("No frame decoded.")
                break
            cv2.imshow("Emotion Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Exited.")
