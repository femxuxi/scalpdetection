import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from model import process_image

app = Flask(__name__)

# Set folder for uploaded images
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowable file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Check if file extension is valid
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

        # Process image for scalp detection
        image = cv2.imread(file_path)
        condition = process_image(image)

        return render_template("result.html", condition=condition, image_url=file_path)
    
    return "Invalid file type", 400

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
