# # camera.py
# """
# Threaded webcam reader + emotion predictor.

# Provides:
# - class WebcamVideoStream: threaded cv2.VideoCapture wrapper (.start(), .read(), .stop())
# - class VideoCamera:
#     - get_frame()      -> (jpeg_bytes, pandas.DataFrame)  # streaming + smoothing
#     - capture_once()   -> (jpeg_bytes, pandas.DataFrame)  # one-shot: prediction from single frame only
#     - stop()           -> release camera & stop thread
# - function music_rec() -> DataFrame of top 15 songs for the current emotion (neutral fallback)
# """

# import time
# import threading
# from threading import Thread
# from collections import deque
# import cv2
# import numpy as np
# from PIL import Image
# import pandas as pd
# import os

# # -------------------------------
# # Model & resources setup
# # -------------------------------
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# # Paths (adjust if needed)
# CASCADE_PATH = "haarcascade_frontalface_default.xml"
# MODEL_PATH = "model.h5"

# # Tuning
# DEFAULT_WINDOW_LEN = 5        # smoothing window for get_frame()
# DEFAULT_CONF_THRESHOLD = 0.55 # confidence threshold used in smoothing (get_frame) and capture_once

# # Emotion labels & CSV mapping (keep consistent with your files)
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
#                 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# music_dist = {
#     0: "songs/angry.csv",
#     1: "songs/disgusted.csv",
#     2: "songs/fearful.csv",
#     3: "songs/happy.csv",
#     4: "songs/neutral.csv",
#     5: "songs/sad.csv",
#     6: "songs/surprised.csv"
# }

# # Build model architecture (same as your original)
# def build_emotion_model():
#     model = Sequential()
#     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
#     model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Flatten())
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(7, activation='softmax'))
#     return model

# # Try to load model weights
# emotion_model = None
# try:
#     emotion_model = build_emotion_model()
#     if os.path.exists(MODEL_PATH):
#         emotion_model.load_weights(MODEL_PATH)
#         print("[camera.py] Loaded emotion model from", MODEL_PATH)
#     else:
#         print("[camera.py] model.h5 not found; continuing without model")
#         emotion_model = None
# except Exception as e:
#     emotion_model = None
#     print("[camera.py] Failed to load emotion model:", e)

# # Load face cascade
# try:
#     face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
#     if face_cascade.empty():
#         raise Exception("Cascade file loaded but is empty")
#     print("[camera.py] Loaded Haar cascade from", CASCADE_PATH)
# except Exception as e:
#     face_cascade = None
#     print("[camera.py] Could not load Haar cascade:", e)

# # disable OpenCL as before
# cv2.ocl.setUseOpenCL(False)

# # -------------------------------
# # Threaded capture class
# # -------------------------------
# class WebcamVideoStream:
#     """
#     Threaded wrapper around cv2.VideoCapture with .start(), .read(), .stop()
#     """
#     def __init__(self, src=0):
#         self.src = src
#         try:
#             # prefer CAP_DSHOW on Windows for performance
#             self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
#             if not self.stream.isOpened():
#                 self.stream = cv2.VideoCapture(src)
#         except Exception:
#             self.stream = cv2.VideoCapture(src)
#         self.grabbed, self.frame = self.stream.read()
#         self.stopped = False
#         self.lock = threading.Lock()
#         self.thread = None

#     def start(self):
#         if self.thread is None:
#             self.thread = Thread(target=self.update, daemon=True)
#             self.thread.start()
#         return self

#     def update(self):
#         while not self.stopped:
#             try:
#                 grabbed, frame = self.stream.read()
#                 with self.lock:
#                     self.grabbed = grabbed
#                     self.frame = frame
#             except Exception:
#                 # keep looping; small delay
#                 pass
#             time.sleep(0.01)

#     def read(self):
#         with self.lock:
#             if self.frame is None:
#                 return None
#             # return a safe copy
#             return self.frame.copy()

#     def stop(self):
#         self.stopped = True
#         if self.thread is not None:
#             self.thread.join(timeout=1.0)
#             self.thread = None
#         try:
#             if hasattr(self.stream, "release"):
#                 self.stream.release()
#         except Exception:
#             pass

# # -------------------------------
# # VideoCamera wrapper
# # -------------------------------
# class VideoCamera(object):
#     """
#     Wrapper exposing get_frame(), capture_once(), stop().
#     get_frame() returns streaming frames and uses smoothing across frames.
#     capture_once() predicts only from that single captured frame (no smoothing).
#     """
#     def __init__(self, src=0, window_len=DEFAULT_WINDOW_LEN, conf_threshold=DEFAULT_CONF_THRESHOLD):
#         self.src = src
#         self.cap = WebcamVideoStream(src=src).start()
#         self.window_len = int(window_len)
#         self.conf_threshold = float(conf_threshold)
#         self.emotion_window = deque(maxlen=self.window_len)
#         self.current_emotion = 4  # Neutral by default
#         self.lock = threading.Lock()

#     def stop(self):
#         """Stop the capture thread and release device."""
#         try:
#             if hasattr(self, 'cap') and self.cap is not None:
#                 self.cap.stop()
#         except Exception:
#             pass

#     def _predict_emotion(self, face_gray):
#         """
#         Predict emotion index from a grayscale face array (48x48).
#         Returns (index, confidence) or (None, 0.0) if model missing/fails.
#         """
#         if emotion_model is None:
#             return None, 0.0
#         try:
#             face_resized = cv2.resize(face_gray, (48, 48))
#             arr = face_resized.astype('float32') / 255.0
#             arr = np.expand_dims(arr, axis=0)
#             arr = np.expand_dims(arr, axis=-1)  # (1,48,48,1)
#             preds = emotion_model.predict(arr, verbose=0)[0]
#             idx = int(np.argmax(preds))
#             conf = float(np.max(preds))
#             return idx, conf
#         except Exception:
#             return None, 0.0

#     def _read_recs_for_emotion(self, emotion_index, top_n=15):
#         """Load and normalize songs CSV for given emotion index."""
#         try:
#             csv_path = music_dist.get(int(emotion_index))
#             if csv_path is None or not os.path.exists(csv_path):
#                 return pd.DataFrame(columns=["Name", "Album", "Artist"])
#             df = pd.read_csv(csv_path)
#             cols = df.columns.tolist()
#             rename_map = {}
#             if "Name" not in cols and len(cols) > 0:
#                 rename_map[cols[0]] = "Name"
#             if "Album" not in cols and len(cols) > 1:
#                 rename_map[cols[1]] = "Album"
#             if "Artist" not in cols and len(cols) > 2:
#                 rename_map[cols[2]] = "Artist"
#             if rename_map:
#                 df = df.rename(columns=rename_map)
#             df = df.reindex(columns=["Name", "Album", "Artist"])
#             return df.head(top_n)
#         except Exception:
#             return pd.DataFrame(columns=["Name", "Album", "Artist"])

#     def get_frame(self):
#         """
#         Streaming method used by video_feed.
#         Uses smoothing (emotion_window) and updates self.current_emotion.
#         Returns (jpeg_bytes, DataFrame) where DataFrame corresponds to the smoothed emotion.
#         """
#         frame = self.cap.read()
#         if frame is None:
#             blank = np.zeros((480, 640, 3), dtype=np.uint8)
#             ret, jpeg = cv2.imencode('.jpg', blank)
#             return jpeg.tobytes(), pd.DataFrame(columns=["Name", "Album", "Artist"])

#         disp = cv2.resize(frame, (640, 480))
#         df_local = self._read_recs_for_emotion(self.current_emotion)

#         if face_cascade is not None:
#             gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#             if len(faces) == 0:
#                 # clear smoothing to avoid stale predictions if desired
#                 with self.lock:
#                     self.emotion_window.clear()
#             else:
#                 faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
#                 x, y, w, h = faces[0]
#                 cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 face_gray = gray[y:y + h, x:x + w]
#                 pred_idx, conf = self._predict_emotion(face_gray)
#                 if pred_idx is not None:
#                     if conf >= self.conf_threshold:
#                         with self.lock:
#                             self.emotion_window.append(pred_idx)
#                     with self.lock:
#                         try:
#                             final_idx = max(set(self.emotion_window), key=self.emotion_window.count) if len(self.emotion_window) > 0 else pred_idx
#                         except Exception:
#                             final_idx = pred_idx
#                         self.current_emotion = int(final_idx)
#                     label = emotion_dict.get(self.current_emotion, str(self.current_emotion))
#                     cv2.putText(disp, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
#                     df_local = self._read_recs_for_emotion(self.current_emotion)
#                 else:
#                     # if model unavailable, keep previous emotion
#                     pass

#         try:
#             ret, jpeg = cv2.imencode('.jpg', disp)
#             jpeg_bytes = jpeg.tobytes()
#         except Exception:
#             blank = np.zeros((480, 640, 3), dtype=np.uint8)
#             ret, jpeg = cv2.imencode('.jpg', blank)
#             jpeg_bytes = jpeg.tobytes()

#         return jpeg_bytes, df_local

#     def capture_once(self):
#         """
#         One-shot capture: predict emotion only from this single frame.
#         Does NOT modify smoothing window or self.current_emotion.
#         Returns (jpeg_bytes, DataFrame) based solely on this captured image.
#         """
#         # Attempt to read a single frame
#         try:
#             frame = self.cap.read()
#         except Exception:
#             frame = None

#         if frame is None:
#             blank = np.zeros((480, 640, 3), dtype=np.uint8)
#             ret, jpeg = cv2.imencode('.jpg', blank)
#             return jpeg.tobytes(), pd.DataFrame(columns=["Name", "Album", "Artist"])

#         disp = cv2.resize(frame, (640, 480))
#         df_local = self._read_recs_for_emotion(self.current_emotion)  # fallback

#         if face_cascade is not None:
#             gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#             if len(faces) > 0:
#                 faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
#                 x, y, w, h = faces[0]
#                 # draw rect for preview
#                 cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 face_gray = gray[y:y + h, x:x + w]
#                 pred_idx, conf = self._predict_emotion(face_gray)
#                 if pred_idx is not None:
#                     # use prediction from this frame only (you can require conf >= threshold if desired)
#                     chosen_idx = int(pred_idx) if conf >= 0.0 else int(pred_idx)
#                     label = emotion_dict.get(chosen_idx, str(chosen_idx))
#                     cv2.putText(disp, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
#                     df_local = self._read_recs_for_emotion(chosen_idx)
#                 else:
#                     # model missing/failure: keep fallback df_local
#                     pass
#             else:
#                 # no face found: keep fallback df_local
#                 pass
#         else:
#             # no cascade: df_local remains fallback
#             pass

#         try:
#             ret, jpeg = cv2.imencode('.jpg', disp)
#             jpeg_bytes = jpeg.tobytes()
#         except Exception:
#             blank = np.zeros((480, 640, 3), dtype=np.uint8)
#             ret, jpeg = cv2.imencode('.jpg', blank)
#             jpeg_bytes = jpeg.tobytes()

#         return jpeg_bytes, df_local

# # -------------------------------
# # Backwards-compatible helper
# # -------------------------------
# def music_rec():
#     """
#     Return top 15 songs for the current emotion (neutral fallback).
#     """
#     try:
#         df = pd.read_csv(music_dist.get(4))
#         df = df.reindex(columns=["Name", "Album", "Artist"]).head(15)
#         return df
#     except Exception:
#         return pd.DataFrame(columns=["Name", "Album", "Artist"])
# camera.py
"""
Threaded webcam reader + emotion predictor (updated)

Changes in this version:
- Uses tensorflow.keras consistently.
- Tries to load full model via tf.keras.models.load_model(). If that fails, falls back to building the architecture and loading weights.
- Removes any training code; this file is inference-only.
- Clearer logging and robust file-path handling.
- Keeps original threaded webcam and smoothing behavior.
"""

import time
import threading
from threading import Thread
from collections import deque
import cv2
import numpy as np
import pandas as pd
import os

# -------------------------------
# Model & resources setup
# -------------------------------
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Paths (adjust if needed)
CASCADE_PATH = "haarcascade_frontalface_default.xml"
MODEL_PATH = "model2.h5"  # change to your saved model file (full model or weights-only)

# Tuning
DEFAULT_WINDOW_LEN = 5        # smoothing window for get_frame()
DEFAULT_CONF_THRESHOLD = 0.55 # confidence threshold used in smoothing (get_frame) and capture_once

# Emotion labels & CSV mapping (keep consistent with your files)
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

# Build model architecture (must match training architecture if you load weights-only)
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

# Try to load model: prefer full-model load; fall back to weights into built model
emotion_model = None
if os.path.exists(MODEL_PATH):
    try:
        # Try loading the full model (recommended if you used model.save(...))
        emotion_model = load_model(MODEL_PATH)
        print(f"[camera.py] Loaded full model from {MODEL_PATH}")
    except Exception as e_full:
        print(f"[camera.py] load_model failed: {e_full}\nTrying to load weights into built model...")
        try:
            emotion_model = build_emotion_model()
            emotion_model.load_weights(MODEL_PATH)
            print(f"[camera.py] Loaded weights into built model from {MODEL_PATH}")
        except Exception as e_weights:
            print(f"[camera.py] Failed to load weights: {e_weights}")
            emotion_model = None
else:
    print(f"[camera.py] Model file not found at {MODEL_PATH}; running without model")

# Load face cascade
try:
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise Exception("Cascade file loaded but is empty")
    print(f"[camera.py] Loaded Haar cascade from {CASCADE_PATH}")
except Exception as e:
    face_cascade = None
    print(f"[camera.py] Could not load Haar cascade: {e}")

# disable OpenCL as before
cv2.ocl.setUseOpenCL(False)

# -------------------------------
# Threaded capture class
# -------------------------------
class WebcamVideoStream:
    """
    Threaded wrapper around cv2.VideoCapture with .start(), .read(), .stop()
    """
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
        self.current_emotion = 4  # Neutral by default
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
                    chosen_idx = int(pred_idx) if conf >= 0.0 else int(pred_idx)
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
    """
    Return top 15 songs for the current emotion (neutral fallback).
    """
    try:
        df = pd.read_csv(music_dist.get(4))
        df = df.reindex(columns=["Name", "Album", "Artist"]).head(15)
        return df
    except Exception:
        return pd.DataFrame(columns=["Name", "Album", "Artist"])
