import os
import cv2
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from model import process_image, preprocess_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No image uploaded", 400
    
    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Process the image
        condition, processed_images = preprocess_image(file_path)

        # Save processed images
        processed_paths = {}
        for step, img in processed_images.items():
            output_path = os.path.join(app.config["PROCESSED_FOLDER"], f"{step}_{filename}")
            cv2.imwrite(output_path, img)
            processed_paths[step] = output_path

        return render_template("result.html", condition=condition, image_url=file_path, processed_paths=processed_paths)
    
    return "Invalid file type", 400

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    app.run(debug=True)
