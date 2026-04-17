from flask import Flask, request, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    # 🔥 Read image safely
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # ❌ Error handling
    if img is None:
        return "❌ Error: Invalid image file"

    if img.size == 0:
        return "❌ Error: Empty image"

    # Resize
    img = cv2.resize(img, (28, 28))

    # Blur
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Threshold
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 🔥 Centering
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (28, 28))

    # Normalize
    img = img / 255.0

    # Reshape
    img = img.reshape(1, 28, 28, 1)

    # Predict
    pred = model.predict(img)
    result = chr(np.argmax(pred) + 65)

    return f"Prediction: {result}"

if __name__ == "__main__":
    app.run(debug=True, port=5050)