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

        # Apply image processing steps
        condition, processed_images = preprocess_image(file_path)

        # Save processed images
        processed_paths = {}
        for step, img in processed_images.items():
            output_path = os.path.join(app.config["PROCESSED_FOLDER"], f"{step}_{filename}")
            cv2.imwrite(output_path, img)
            processed_paths[step] = output_path

        return render_template("result.html", condition=condition, image_url=file_path, processed_paths=processed_paths)
    
    return "Invalid file type", 400
