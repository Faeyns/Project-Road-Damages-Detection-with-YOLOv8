import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
import tensorflow as tf
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    send_file,
    url_for,
    Response,
)
from werkzeug.utils import secure_filename
import os
import time
from ultralytics import YOLO

app = Flask(__name__)

# Path model terbaik
best_model_path = r"C:\Users\fatur\Documents\Web-app_VK_Road Damages Object Detection\Road Damages Object Detection Web App\yolo-Weights\best.pt"

@app.route("/")
def hello_world():
    if "image_path" in request.args:
        image_path = request.args["image_path"]
        return render_template("index.html", image_path=image_path)
    return render_template("index.html")

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        # Periksa apakah form 'file' ada dan apakah file telah diunggah
        if 'file' not in request.files or request.files['file'].filename == '':
            # Tampilkan pesan kesalahan jika tidak ada file yang diunggah
            return render_template(
                "index.html",
                image_path="",
                media_type='image',
                error_message="No file uploaded. Please select a file to upload."
            )

        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Pastikan folder upload ada
        f.save(filepath)

        file_extension = f.filename.rsplit('.', 1)[1].lower()

        # Handle image upload for .jpg and .png files
        if file_extension in ['jpg', 'png']:
            img = cv2.imread(filepath)

            # Perform the detection
            model = YOLO(best_model_path)
            detections = model.predict(source=filepath, save=True, conf=0.5)

            # Path to 'runs/detect'
            detect_folder_path = r"C:\Users\fatur\runs\detect"
            subfolders = [
                f for f in os.listdir(detect_folder_path)
                if os.path.isdir(os.path.join(detect_folder_path, f))
            ]

            if subfolders:
                # Find the latest subfolder
                latest_subfolder = max(
                    subfolders,
                    key=lambda x: os.path.getctime(os.path.join(detect_folder_path, x))
                )
                detected_image_path = os.path.join(
                    detect_folder_path, latest_subfolder, f.filename
                )

                return render_template(
                    'index.html',
                    image_path=detected_image_path,
                    media_type='image'
                )
            else:
                return render_template(
                    "index.html",
                    image_path="",
                    media_type='image',
                    error_message="No detection results found."
                )

        elif file_extension == "mp4":
            # Handle video upload
            video_path = filepath
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (frame_width, frame_height))

            model = YOLO(best_model_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, save=True)
                res_plotted = results[0].plot()
                out.write(res_plotted)
                if cv2.waitKey(1) == ord("q"):
                    break

            cap.release()
            out.release()
            return render_template(
                'index.html',
                video_path="output.mp4",
                media_type='video'
            )

    return render_template("index.html", image_path="", media_type='image')

@app.route("/<path:filename>")
def display(filename):
    folder_path = r"C:\Users\fatur\runs\detect"
    subfolders = [
        f for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ]

    if subfolders:
        latest_subfolder = max(
            subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x))
        )
        directory = os.path.join(folder_path, latest_subfolder)
        files = os.listdir(directory)
        latest_file = files[0]

        image_path = os.path.join(directory, latest_file)
        file_extension = latest_file.rsplit(".", 1)[1].lower()

        if file_extension == "jpg":
            return send_file(image_path, mimetype="image/jpeg")
        elif file_extension == "mp4":
            return send_file(image_path, mimetype="video/mp4")
        elif file_extension == "png":
            return send_file(image_path, mimetype="image/png")
        else:
            return "Invalid file format"
    else:
        return "No detection results found."

def get_frame():
    mp4_files = "output.mp4"
    video = cv2.VideoCapture(mp4_files)
    while True:
        success, frame = video.read()
        if not success:
            break
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")
        time.sleep(0.1)

@app.route("/video_feed")
def video_feed():
    return Response(get_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/webcam_feed")
def webcam_feed():
    cap = cv2.VideoCapture(0)

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break
            model = YOLO(best_model_path)
            results = model(frame, save=True)
            res_plotted = results[0].plot()
            ret, buffer = cv2.imencode(".jpg", res_plotted)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = YOLO(best_model_path)
    app.run(host="0.0.0.0", port=args.port, debug=True)
