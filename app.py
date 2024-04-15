from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

model = load_model('ch.keras')

def preprocess_image(image_bytes):
  img = Image.open(io.BytesIO(image_bytes))

  img = img.convert('RGB')

  img = img.resize((150, 150))
  img_array = img_to_array(img)
  img_array = img_array / 255.0
  return np.expand_dims(img_array, axis=0)

def predict(image_bytes):
  preprocessed_image = preprocess_image(image_bytes)
  predictions = model.predict(preprocessed_image)
  predicted_class = np.argmax(predictions[0])
  class_names = ['COVID-19', 'Normal', 'Pneumonia']
  return class_names[predicted_class], predictions[0][predicted_class]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def make_prediction():

  if 'image' not in request.files:
    return jsonify({'error': 'No image uploaded'}), 400
  image_bytes = request.files['image'].read()


  prediction, confidence = predict(image_bytes)

  confidence = float(confidence)
  return jsonify({'class': prediction, 'confidence': confidence})

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000)
