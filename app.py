from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("mobilenet_pneumonia.keras")

IMG_SIZE = 224

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['Post'])
def predict():

    file = request.files['file']

    file_path = os.path.join('uploads', file.filename)

    file.save(file_path)

    img = image.load_img(
        file_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis = 0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5 :
        result = "PNEUMONIA"
    else:
        result = "NORMAL"

    confidence = round(float(prediction) * 100 , 2)

    return render_template(
        'index.html',
        prediction=result,
        confidence = confidence
    )

if __name__ == '__main__':
    app.run(debug=True)