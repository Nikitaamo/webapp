import os
import numpy as np
import tarfile
import gdown
import pickle
from flask import Flask, request, render_template_string
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors
from werkzeug.utils import secure_filename

# Setup SSL context to bypass certificate verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Download and extract dataset
def download_and_extract_dataset(url, output_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
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
        upload_folder = './uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        img_path = os.path.join(upload_folder, filename)
        try:
            file.save(img_path)
        except Exception as e:
            return f"Error saving the file: {str(e)}"
        features = extract_features(model_finetuned, img_path)
        distances, indices = neighbors.kneighbors([features])
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
    dataset_path = './datasets/caltech101'
    features_path = 'features-caltech101-resnet.pickle'
    filenames_path = 'filenames-caltech101.pickle'
    model_path = 'model-finetuned.h5'

    download_and_extract_dataset(dataset_url, 'caltech101.tar.gz', dataset_path)

    if os.path.exists(features_path) and os.path.exists(filenames_path) and os.path.exists(model_path):
        feature_list = pickle.load(open(features_path, 'rb'))
        file_list = pickle.load(open(filenames_path, 'rb'))
        model_finetuned = load_model(model_path)
    else:
        print("Some required files are missing. Checking what's available and taking action.")
        if not os.path.exists(model_path):
            print("Model file not found. Please ensure the model has been trained and saved.")
            model_finetuned = initialize_model()  # Optionally retrain model here if that's an option
        else:
            model_finetuned = load_model(model_path)

        if not os.path.exists(dataset_path):
            print("Dataset not found at", dataset_path, "Please check the dataset URL or path.")
        else:
            file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]
            if file_list:
                feature_list = [extract_features(model_finetuned, f) for f in file_list]
                pickle.dump(feature_list, open(features_path, 'wb'))
                pickle.dump(file_list, open(filenames_path, 'wb'))
            else:
                print("No image files found in", dataset_path)

    if 'feature_list' in locals() and feature_list:
        neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(feature_list)
    else:
        print("Feature list is empty, cannot fit NearestNeighbors.")

    app.run(debug=True)

