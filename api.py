from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import os

app = Flask(__name__)

# Load the pre-trained model
model_path = "SoilNet_93_86.h5"
SoilNet = load_model(model_path)

# Define soil classes
classes = {
    0: "Alluvial Soil:-{ Rice, Wheat, Sugarcane, Maize, Cotton, Soyabean, Jute }",
    1: "Black Soil:-{ Virginia, Wheat, Jowar, Millets, Linseed, Castor, Sunflower }",
    2: "Clay Soil:-{ Rice, Lettuce, Chard, Broccoli, Cabbage, Snap Beans }",
    3: "Red Soil:{ Cotton, Wheat, Pilses, Millets, OilSeeds, Potatoes }"
}

# Define API key
api_key = "sih-project"


def model_predict(image_path, model):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)

    result = np.argmax(model.predict(image))
    prediction = classes[result]

    return {'predicted_class': int(result), 'class_label': prediction}


@app.route('/predict', methods=['POST'])
def predict():
    # Check for the API key in the request headers
    if 'X-API-KEY' not in request.headers or request.headers['X-API-KEY'] != api_key:
        return jsonify({'error': 'Invalid API key'}), 401

    # Check if the 'image' file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join('static/user uploaded', filename)
    file.save(file_path)

    try:
        prediction = model_predict(file_path, SoilNet)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(threaded=False)
