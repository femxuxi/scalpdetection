import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(image_path):
    image = cv2.imread(image_path)

    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    noise_reduced = cv2.GaussianBlur(gray, (5,5), 0)

    # Sharpening
    sharpened = cv2.addWeighted(gray, 1.5, noise_reduced, -0.5, 0)

    # Negative/Inverted Imaging
    negative = cv2.bitwise_not(gray)

    # Segmentation (Simple thresholding for affected regions)
    _, segmented = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Resize for model processing
    resized = cv2.resize(gray, (128, 128))

    # Load trained model and predict scalp condition
    model = tf.keras.models.load_model("scalp_model.h5")
    prediction = model.predict(np.expand_dims(resized, axis=0))
    
    # Mapping conditions
    conditions = ["Dandruff", "Normal Scalp", "Folliculitis", "Psoriasis", "Dissecting Cellulitis", "Acne Keloidalis Nuchae"]
    
    # Store processed steps for display
    processed_images = {
        "Grayscale": gray,
        "Noise_Reduction": noise_reduced,
        "Sharpening": sharpened,
        "Negative_Image": negative,
        "Segmentation": segmented
    }

    return conditions[np.argmax(prediction)], processed_images
