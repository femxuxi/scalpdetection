from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from grayscaled.grayscaled import process_and_save_image as grayscale_image
from denoised.noisereduction import process_and_save_image as denoise_image
from segmented.imagesegmentation import segment_image
from sharpened.imagesharpening import sharpen_image
from negative.negativeimages import create_negative_image
from PIL import Image

app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        processed_images = {}
        original_image = Image.open(filepath)

        # Grayscale
        _, gray_image, _ = grayscale_image(filepath)
        gray_path = os.path.join(app.config['PROCESSED_FOLDER'], f'gray_{filename}')
        Image.fromarray(gray_image).save(gray_path)
        processed_images['grayscale'] = gray_path

        # Noise Reduction
        _, denoised_image = denoise_image(filepath)
        denoised_path = os.path.join(app.config['PROCESSED_FOLDER'], f'denoised_{filename}')
        Image.fromarray((denoised_image * 255).astype('uint8')).save(denoised_path)
        processed_images['denoised'] = denoised_path

        # Segmentation
        _, segmented_image = segment_image(filepath)
        segmented_path = os.path.join(app.config['PROCESSED_FOLDER'], f'segmented_{filename}')
        segmented_image.save(segmented_path)
        processed_images['segmented'] = segmented_path

        # Sharpening
        sharpened_image = sharpen_image(original_image)
        sharpened_path = os.path.join(app.config['PROCESSED_FOLDER'], f'sharpened_{filename}')
        sharpened_image.save(sharpened_path)
        processed_images['sharpened'] = sharpened_path

        # Negative
        _, negative_image = create_negative_image(filepath)
        negative_path = os.path.join(app.config['PROCESSED_FOLDER'], f'negative_{filename}')
        negative_image.save(negative_path)
        processed_images['negative'] = negative_path

        # Detect scalp condition (placeholder logic)
        scalp_condition = "Healthy Scalp"
        if 'dry' in filename.lower():
            scalp_condition = "Dry Scalp"
        elif 'oily' in filename.lower():
            scalp_condition = "Oily Scalp"

        return jsonify({
            'status': 'success',
            'processed_images': processed_images,
            'scalp_condition': scalp_condition
        })

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/processed/<filename>')
def get_processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
