# app.py
import base64
import time
import threading
from pathlib import Path

from flask import Flask, render_template, Response, jsonify, request
import pandas as pd

# Try to import VideoCamera from camera.py (the updated camera module)
try:
    from camera import VideoCamera
    _HAS_CAMERA = True
except Exception as e:
    VideoCamera = None
    _HAS_CAMERA = False
    print("[app.py] Warning: camera module not available:", e)

# Try to import emoji backend helper (maps emoji -> DataFrame)
try:
    from emoji_backend import emoji_to_df
except Exception:
    def emoji_to_df(k, top_n=15):
        # fallback: return empty DataFrame with expected columns
        return pd.DataFrame(columns=["Name", "Album", "Artist"])

app = Flask(__name__)

# UI table headings
headings = ("Name", "Album", "Artist")

# Application state (simple globals for single-user demo)
_state_lock = threading.Lock()
current_mode = "emoji"   # "emoji" or "camera"
current_emoji = "😐"
df1 = emoji_to_df(current_emoji).head(15)

# Global camera instance (lazily created)
_camera_instance = None


def get_or_create_camera():
    """Lazily create and return the global VideoCamera instance, or None if not available."""
    global _camera_instance
    if not _HAS_CAMERA:
        return None
    if _camera_instance is None:
        try:
            _camera_instance = VideoCamera()
            # small warmup for camera threads
            print("[app.py] Created VideoCamera instance")
            time.sleep(0.12)
        except Exception as e:
            _camera_instance = None
            app.logger.exception("Failed to create VideoCamera: %s", e)
    return _camera_instance


def release_camera():
    """
    Safely stop and release the global camera instance so the OS device is freed.
    Expects VideoCamera.stop() to exist (the updated camera.py provides it).
    """
    global _camera_instance
    if _camera_instance is None:
        return
    try:
        print("[app.py] Releasing camera...")
        if hasattr(_camera_instance, "stop") and callable(_camera_instance.stop):
            try:
                _camera_instance.stop()
            except Exception:
                app.logger.exception("Error calling VideoCamera.stop()")
        # defensive: try inner cap.stop()
        if hasattr(_camera_instance, "cap") and hasattr(_camera_instance.cap, "stop") and callable(_camera_instance.cap.stop):
            try:
                _camera_instance.cap.stop()
            except Exception:
                app.logger.exception("Error calling cap.stop()")
    except Exception:
        app.logger.exception("Unexpected error while releasing camera")
    finally:
        _camera_instance = None
        print("[app.py] Camera released.")


def gen(camera):
    """
    Generator that yields multipart JPEG frames from camera.get_frame().
    camera.get_frame() must return (jpeg_bytes, optional_dataframe).
    If a dataframe is returned it's used to update the global df1.
    """
    global df1
    while True:
        try:
            frame_bytes, new_df = camera.get_frame()
        except Exception as e:
            app.logger.exception("camera.get_frame() error: %s", e)
            # yield a blank jpeg fragment to keep client alive
            blank = (b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff' + b'\r\n\r\n')
            yield blank
            time.sleep(0.1)
            continue

        # update global recommendations if camera returned a DataFrame
        if isinstance(new_df, pd.DataFrame):
            with _state_lock:
                try:
                    cleaned = new_df.copy()
                    expected = ["Name", "Album", "Artist"]
                    if not all(col in cleaned.columns for col in expected):
                        cols = list(cleaned.columns)
                        mapping = {}
                        if len(cols) > 0:
                            mapping[cols[0]] = "Name"
                        if len(cols) > 1:
                            mapping[cols[1]] = "Album"
                        if len(cols) > 2:
                            mapping[cols[2]] = "Artist"
                        if mapping:
                            cleaned = cleaned.rename(columns=mapping)
                    cleaned = cleaned.reindex(columns=expected)
                    df1 = cleaned.head(15)
                except Exception:
                    app.logger.exception("Failed to sanitize dataframe from camera; keeping previous df1")

        # yield frame bytes as multipart
        try:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        except Exception:
            app.logger.exception("Failed to yield frame bytes")
            time.sleep(0.05)


# @app.route('/')
# def index():
#     """Render main page (index.html). Template expects headings, data, current_mode."""
#     with _state_lock:
#         local_df = df1.copy() if isinstance(df1, pd.DataFrame) else pd.DataFrame(columns=["Name","Album","Artist"])
#         mode = current_mode
#     return render_template('index.html', headings=headings, data=local_df, current_mode=mode)

# @app.route('/get-started')
# def get_started():
#     return render_template('get_started.html')

@app.route('/')
def landing():
    return render_template('get_started.html')

@app.route('/app')
def index():
    with _state_lock:
        local_df = df1.copy() if isinstance(df1, pd.DataFrame) else pd.DataFrame(columns=["Name","Album","Artist"])
        mode = current_mode
    return render_template('index.html', headings=headings, data=local_df, current_mode=mode)


@app.route('/video_feed')
def video_feed():
    """Stream video. Create camera lazily; if unavailable return 503."""
    cam = get_or_create_camera()
    if cam is None:
        return ("Camera not available", 503)
    return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_recs')
def get_recs():
    """Return current recommendations as JSON (alias /t also provided)."""
    with _state_lock:
        local_df = df1.copy() if isinstance(df1, pd.DataFrame) else pd.DataFrame(columns=["Name","Album","Artist"])
    return local_df.to_json(orient='records')


@app.route('/t')
def gen_table():
    return get_recs()


@app.route('/set_emoji', methods=['POST'])
def set_emoji():
    """Set emoji mode and update recommendations based on emoji-to-CSV mapping."""
    global current_mode, current_emoji, df1
    data = request.get_json(silent=True) or {}
    emoji = data.get('emoji') or request.form.get('emoji') or request.args.get('emoji')
    if not emoji:
        return jsonify({"status": "error", "message": "No emoji provided"}), 400
    try:
        new_df = emoji_to_df(emoji, top_n=15)
        if not isinstance(new_df, pd.DataFrame):
            new_df = pd.DataFrame(new_df)
        with _state_lock:
            df1 = new_df.reindex(columns=["Name","Album","Artist"]).head(15)
            current_emoji = emoji
            current_mode = "emoji"
        # when switching to emoji mode, release camera resources
        release_camera()
        return jsonify({"status": "ok", "message": "emoji set", "data": df1.to_dict(orient='records')})
    except Exception as e:
        app.logger.exception("Failed to set emoji: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/set_mode', methods=['POST'])
def set_mode():
    """
    Switch between 'camera' and 'emoji' modes. JSON: {'mode':'camera'|'emoji'}.
    Releases camera when switching to emoji.
    """
    global current_mode
    data = request.get_json(silent=True) or {}
    mode = data.get('mode') or request.form.get('mode') or request.args.get('mode')
    if mode not in ("camera", "emoji"):
        return jsonify({"status": "error", "message": "mode must be 'camera' or 'emoji'"}), 400

    if mode == "camera":
        cam = get_or_create_camera()
        if cam is None:
            return jsonify({"status": "error", "message": "camera not available on server"}), 503
    else:
        # emoji -> release camera to fully free device
        try:
            release_camera()
        except Exception:
            app.logger.exception("release_camera failed")

    with _state_lock:
        current_mode = mode
    return jsonify({"status": "ok", "mode": current_mode})

@app.route('/capture_face', methods=['POST'])
def capture_face():
    """
    One-shot capture endpoint: capture a single frame, get recommendations based only on that frame,
    return base64 image and recommendations. After capture the camera is released so device is freed.
    """
    cam = get_or_create_camera()
    if cam is None:
        return jsonify({"status": "error", "message": "Camera not available"}), 503

    try:
        # Use capture_once() if available (predicts only from this frame)
        if hasattr(cam, "capture_once") and callable(cam.capture_once):
            frame_bytes, new_df = cam.capture_once()
        else:
            # fallback: call get_frame() but it might include smoothing / previous results
            frame_bytes, new_df = cam.get_frame()

        global df1, current_mode
        if isinstance(new_df, pd.DataFrame):
            with _state_lock:
                try:
                    cleaned = new_df.copy()
                    cleaned = cleaned.reindex(columns=["Name","Album","Artist"])
                    df1 = cleaned.head(15)
                except Exception:
                    app.logger.exception("Failed to sanitize new_df in capture_face")

        # encode image bytes to base64 for inline preview
        img_b64 = base64.b64encode(frame_bytes).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_b64}"

        # After capture, release camera to fully turn it off
        try:
            release_camera()
        except Exception:
            app.logger.exception("release_camera failed after capture")
        with _state_lock:
            current_mode = "emoji"

        with _state_lock:
            out_df = df1.copy() if isinstance(df1, pd.DataFrame) else pd.DataFrame(columns=["Name","Album","Artist"])
        return jsonify({
            "status": "ok",
            "image": img_data_url,
            "data": out_df.to_dict(orient='records')
        })
    except Exception as e:
        app.logger.exception("capture_face failed: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.debug = True
    # bind to all interfaces by default (change if you prefer local only)
    app.run(host='0.0.0.0', port=5000)
