def process_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply noise reduction
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Enhance details with sharpening
    sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # Negative image for contrast boost
    negative = cv2.bitwise_not(sharpened)

    # Resize for model processing
    resized = cv2.resize(negative, (128, 128))

    # Load trained model and predict condition
    model = tf.keras.models.load_model("scalp_model.h5")
    prediction = model.predict(np.expand_dims(resized, axis=0))

    # Scalp condition labels
    conditions = ["Dandruff", "Normal Scalp", "Folliculitis", "Psoriasis", "Dissecting Cellulitis", "Acne Keloidalis Nuchae"]
    return conditions[np.argmax(prediction)]
