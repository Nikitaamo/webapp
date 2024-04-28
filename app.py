import os
import numpy as np
import tarfile
import gdown
from flask import Flask, request, render_template_string
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors
from werkzeug.utils import secure_filename

# Download and extract dataset
def download_and_extract_dataset(url, output_path, extract_to):
    gdown.download(url, output_path, quiet=False)
    with tarfile.open(output_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    os.remove(output_path)  # Cleanup

# Initialize the model
def initialize_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Extract features from an image
def extract_features(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    return features.flatten()

# Setup Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        img_path = os.path.join('./uploads', filename)
        file.save(img_path)
        features = extract_features(model, img_path)
        distances, indices = knn.kneighbors([features])
        return render_template_string('''
            <h1>Similar Images</h1>
            {% for index in indices[0] %}
            <img src="{{ file_list[index] }}" width="100" height="100">
            {% endfor %}
        ''', indices=indices, file_list=file_list)
    return '''
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit">
    </form>
    '''

# Main script execution
if __name__ == '__main__':
    dataset_url = 'https://drive.google.com/uc?id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp'
    download_and_extract_dataset(dataset_url, 'caltech101.tar.gz', './datasets')
    model = initialize_model()
    file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk('./datasets/caltech101') for f in filenames if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]
    features_list = [extract_features(model, f) for f in file_list]
    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(features_list)
    app.run(debug=True)
