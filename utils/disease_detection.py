from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('models/plant_disease_model.h5')

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

@app.route('/disease-detect', methods=['POST'])
def disease_detect():
    if request.method == 'POST':
        file = request.files['plant_image']
        file_path = f'static/uploads/{file.filename}'
        file.save(file_path)

        # Make prediction
        img = prepare_image(file_path)
        prediction = model.predict(img)
        result = np.argmax(prediction, axis=1)

        diseases = ['Healthy', 'Rust', 'Blight']  # Modify this according to your model
        disease_name = diseases[result[0]]
        return render_template('disease_result.html', disease=disease_name)

if __name__ == '__main__':
    app.run(debug=True)
