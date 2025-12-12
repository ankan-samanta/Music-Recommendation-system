# camera.py
"""
Threaded webcam reader + emotion predictor.

Provides:
- class WebcamVideoStream: threaded cv2.VideoCapture wrapper (.start(), .read(), .stop())
- class VideoCamera:
    - get_frame()      -> (jpeg_bytes, pandas.DataFrame)
    - capture_once()   -> (jpeg_bytes, pandas.DataFrame)
    - stop()           -> release camera & stop thread
- function music_rec() -> DataFrame of top 15 songs for the current emotion (neutral fallback)
"""

import time
import threading
from threading import Thread
from collections import deque
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
import traceback

# -------------------------------
# Model & resources setup
# -------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model

# Paths (adjust if needed)
CASCADE_PATH = "haarcascade_frontalface_default.xml"
MODEL_PATH = "model3.h5"   # <-- change to the filename you actually have

# Tuning
DEFAULT_WINDOW_LEN = 5
DEFAULT_CONF_THRESHOLD = 0.55

# Emotion labels & CSV mapping
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

music_dist = {
    0: "songs/angry.csv",
    1: "songs/disgusted.csv",
    2: "songs/fearful.csv",
    3: "songs/happy.csv",
    4: "songs/neutral.csv",
    5: "songs/sad.csv",
    6: "songs/surprised.csv"
}

def build_emotion_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model

# Attempt to load model robustly
emotion_model = None
if os.path.exists(MODEL_PATH):
    # Try loading the full model first
    try:
        emotion_model = load_model(MODEL_PATH)
        print(f"[camera.py] Loaded full model via load_model('{MODEL_PATH}')")
    except Exception as e_load:
        print(f"[camera.py] load_model failed: {e_load}. Trying to load weights into architecture...")
        traceback.print_exc()
        # fallback: build arch and load weights
        try:
            emotion_model = build_emotion_model()
            emotion_model.load_weights(MODEL_PATH)
            print(f"[camera.py] Loaded model weights into built architecture from '{MODEL_PATH}'")
        except Exception as e_w:
            print(f"[camera.py] Failed to load weights into built model: {e_w}")
            traceback.print_exc()
            emotion_model = None
else:
    print(f"[camera.py] MODEL_PATH '{MODEL_PATH}' not found. Running without a trained model.")
    emotion_model = None

# Load face cascade
face_cascade = None
if os.path.exists(CASCADE_PATH):
    try:
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if face_cascade.empty():
            print(f"[camera.py] Cascade file '{CASCADE_PATH}' loaded but is empty.")
            face_cascade = None
        else:
            print(f"[camera.py] Loaded Haar cascade from '{CASCADE_PATH}'")
    except Exception as e:
        print("[camera.py] Could not load cascade:", e)
        face_cascade = None
else:
    print(f"[camera.py] Cascade file '{CASCADE_PATH}' not found.")

cv2.ocl.setUseOpenCL(False)

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
        if emotion_model is None:
            return None, 0.0
        try:
            face_resized = cv2.resize(face_gray, (48, 48))
            arr = face_resized.astype('float32') / 255.0
            arr = np.expand_dims(arr, axis=0)
            arr = np.expand_dims(arr, axis=-1)
            preds = emotion_model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(np.max(preds))
            return idx, conf
        except Exception:
            return None, 0.0

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
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_gray = gray[y:y + h, x:x + w]
                pred_idx, conf = self._predict_emotion(face_gray)
                if pred_idx is not None:
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
                    cv2.putText(disp, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
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
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_gray = gray[y:y + h, x:x + w]
                pred_idx, conf = self._predict_emotion(face_gray)
                if pred_idx is not None:
                    chosen_idx = int(pred_idx)
                    label = emotion_dict.get(chosen_idx, str(chosen_idx))
                    cv2.putText(disp, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    df_local = self._read_recs_for_emotion(chosen_idx)

        try:
            ret, jpeg = cv2.imencode('.jpg', disp)
            jpeg_bytes = jpeg.tobytes()
        except Exception:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, jpeg = cv2.imencode('.jpg', blank)
            jpeg_bytes = jpeg.tobytes()

        return jpeg_bytes, df_local

# -------------------------------
# Backwards-compatible helper
# -------------------------------
def music_rec():
    try:
        df = pd.read_csv(music_dist.get(4))
        df = df.reindex(columns=["Name", "Album", "Artist"]).head(15)
        return df
    except Exception:
        return pd.DataFrame(columns=["Name", "Album", "Artist"])
