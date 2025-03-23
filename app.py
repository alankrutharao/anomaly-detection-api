# import os
# import tempfile
# import cv2
# from flask import Flask, request, send_file, jsonify
# from transformers import pipeline
# from io import BytesIO
# from PIL import Image
# import torch
# from flask_cors import CORS  # Import Flask-CORS for handling cross-origin requests

# app = Flask(__name__)
# CORS(app)  # Enable CORS to allow React to communicate with Flask

# # Check if CUDA (GPU) is available, otherwise use CPU
# device = 0 if torch.cuda.is_available() else -1

# # Initialize the video classification pipeline
# pipe = pipeline("video-classification", model="Sathwik-kom/anomaly-detector-videomae10", device=device)

# def detect_anomalies_in_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     min_frames, num_frames = 4, 16
#     frames, frame_count = [], 0
#     highest_anomaly_score, highlighted_frame, highlighted_timestamp = 0, None, 0
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default FPS to 30 if unavailable
#     frame_skip = max(1, int(fps * 0.2))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_count % frame_skip == 0:
#             frame_resized = cv2.resize(frame, (224, 224))
#             frames.append(frame_resized / 255.0)

#             if len(frames) >= min(min_frames, len(frames)):
#                 while len(frames) < num_frames:
#                     frames.append(frames[-1])

#                 temp_video_path = tempfile.mktemp(suffix=".mp4")
#                 try:
#                     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#                     out = cv2.VideoWriter(temp_video_path, fourcc, fps, (224, 224))

#                     for f in frames[-num_frames:]:
#                         out.write((f * 255).astype("uint8"))
#                     out.release()

#                     result = pipe(temp_video_path)

#                     normal_score = result[0]["score"] if result[0]["label"] == "LABEL_0" else 0
#                     anomalous_score = result[0]["score"] if result[0]["label"] == "LABEL_1" else 0

#                     timestamp = (frame_count - len(frames) + 1) / fps
#                     if anomalous_score > normal_score and anomalous_score > highest_anomaly_score:
#                         highest_anomaly_score = anomalous_score
#                         highlighted_frame = frame.copy()
#                         highlighted_timestamp = timestamp

#                 finally:
#                     try:
#                         os.unlink(temp_video_path)  # Delete the temporary video file
#                     except PermissionError:
#                         print(f"Warning: Could not delete {temp_video_path}, retrying later.")

#                 frames.pop(0)

#         frame_count += 1

#     cap.release()

#     if highlighted_frame is not None:
#         height, width, _ = highlighted_frame.shape
#         cv2.rectangle(highlighted_frame, (0, 0), (width, height), (0, 0, 255), 2)
#         text = f"Anomaly Detected at {highlighted_timestamp:.2f}s, Score: {highest_anomaly_score:.4f}"
#         cv2.putText(
#             highlighted_frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
#         )

#         _, buffer = cv2.imencode(".png", highlighted_frame)
#         return buffer.tobytes(), highlighted_timestamp, highest_anomaly_score
#     else:
#         return None, None, None

# @app.route("/", methods=["GET", "POST"])
# def upload_video():
#     if request.method == "POST":
#         if "video" not in request.files or request.files["video"].filename == "":
#             return jsonify({"message": "No video file uploaded"}), 400

#         video = request.files["video"]
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#             video.save(temp_video.name)
#             try:
#                 anomaly_frame, timestamp, score = detect_anomalies_in_video(temp_video.name)

#                 if anomaly_frame:
#                     # Convert the anomaly frame to image and send as a response
#                     image = Image.open(BytesIO(anomaly_frame))
#                     image_io = BytesIO()
#                     image.save(image_io, "PNG")
#                     image_io.seek(0)

#                     # Return the image as a response to the React frontend
#                     return send_file(
#                         image_io, mimetype="image/png", as_attachment=False, download_name="anomaly_frame.png"
#                     )
#                 else:
#                     return jsonify({"message": "No anomaly detected in the video."}), 200
#             finally:
#                 try:
#                     os.unlink(temp_video.name)
#                 except PermissionError:
#                     print(f"Warning: Could not delete {temp_video.name}, retrying later.")

#     return """
#     <!doctype html>
#     <title>Video Anomaly Detection</title>
#     <h1>Upload a video to detect anomalies</h1>
#     <form method="post" enctype="multipart/form-data">
#         <input type="file" name="video">
#         <input type="submit" value="Upload">
#     </form>
#     """

import os
import tempfile
import gc
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from transformers import pipeline
from io import BytesIO
from PIL import Image
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("video-classification", model="Sathwik-kom/anomaly-detector-videomae10", device=device, torch_dtype=torch.float16 if device == 0 else torch.float32)

def detect_anomalies_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    min_frames, num_frames = 4, 16
    frames, frame_count = [], 0
    highest_anomaly_score, highlighted_frame, highlighted_timestamp = 0, None, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip = max(1, int(fps * 0.2))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_resized = cv2.resize(frame, (224, 224)).astype("float32") / 255.0
            frames.append(frame_resized)
            if len(frames) > num_frames:
                frames.pop(0)

            if len(frames) == num_frames:
                frames_np = np.array(frames).astype("float32")
                result = pipe(frames_np)

                anomalous_score = result[0]["score"] if result[0]["label"] == "LABEL_1" else 0
                timestamp = (frame_count - len(frames) + 1) / fps
                
                if anomalous_score > highest_anomaly_score:
                    highest_anomaly_score = anomalous_score
                    highlighted_frame = frame.copy()
                    highlighted_timestamp = timestamp
        
        frame_count += 1
    
    cap.release()
    del frames
    gc.collect()

    if highlighted_frame is not None:
        height, width, _ = highlighted_frame.shape
        cv2.rectangle(highlighted_frame, (0, 0), (width, height), (0, 0, 255), 2)
        text = f"Anomaly Detected at {highlighted_timestamp:.2f}s, Score: {highest_anomaly_score:.4f}"
        cv2.putText(highlighted_frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        _, buffer = cv2.imencode(".png", highlighted_frame)
        return buffer.tobytes(), highlighted_timestamp, highest_anomaly_score
    else:
        return None, None, None

@app.route("/", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        if "video" not in request.files or request.files["video"].filename == "":
            return jsonify({"message": "No video file uploaded"}), 400

        video = request.files["video"]
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        if video.content_length and video.content_length > MAX_FILE_SIZE:
            return jsonify({"message": "File too large"}), 400

        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
            video.save(temp_video.name)
            anomaly_frame, timestamp, score = detect_anomalies_in_video(temp_video.name)

            if anomaly_frame:
                image = Image.open(BytesIO(anomaly_frame))
                image_io = BytesIO()
                image.save(image_io, "PNG")
                image_io.seek(0)
                return send_file(image_io, mimetype="image/png", as_attachment=False, download_name="anomaly_frame.png")
            else:
                return jsonify({"message": "No anomaly detected in the video."}), 200

    return """
    <!doctype html>
    <title>Video Anomaly Detection</title>
    <h1>Upload a video to detect anomalies</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="video">
        <input type="submit" value="Upload">
    </form>
    """