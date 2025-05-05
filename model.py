import cv2
import numpy as np
import tensorflow as tf

def process_image(image):
    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sharpening
    sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # Load trained model (assuming TensorFlow)
    model = tf.keras.models.load_model("scalp_model.h5")
    resized = cv2.resize(sharpened, (128, 128))
    prediction = model.predict(np.expand_dims(resized, axis=0))

    # Map conditions
    conditions = ["Dandruff", "Normal Scalp", "Folliculitis", "Psoriasis", "Dissecting Cellulitis", "Acne Keloidalis Nuchae"]
    return conditions[np.argmax(prediction)]
