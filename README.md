
# Image Similarity Flask Web Application

This web application uses TensorFlow and the ResNet50 model to find and display images similar to the one uploaded by the user. The backend is built with Flask, and it allows users to upload images and view a list of similar images based on feature extraction and nearest neighbor search.

## Getting Started

These instructions will help you set up the project locally on your machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3 or
- pip
- virtualenv (optional)

### Installation

Clone the repository to your local machine:

git clone https://github.com/your-username/your-repo-name.git

Navigate to the project directory:

cd your-repo-name

Create a virtual environment (optional):

virtualenv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

### Running the Application

Run the Flask application:

python app.py

Open your web browser and go to `http://localhost:5000` to use the application.

### Using the Application

- Click the "Choose File" button to select an image file from your computer.
- Click the "Submit" button to upload the image and view similar images.

## Built With

- [Flask](https://flask.palletsprojects.com/en/2.0.x/) - The web framework used.
- [TensorFlow](https://www.tensorflow.org/) - The machine learning framework used.
- [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function) - The pre-trained model used for feature extraction.
- [NumPy](https://numpy.org/) - Used for numerical operations.
- [OpenCV](https://opencv.org/) - Used for image processing tasks.
- [gdown](https://pypi.org/project/gdown/) - Used for downloading files from Google Drive.

## Lessons Learned

Implementing this solution taught us how to automate the download and extraction of datasets using Python, demonstrating efficient data management for machine learning applications. We utilized the ResNet50 pre-trained model for feature extraction, leveraging the power of transfer learning to efficiently process images without extensive computational resources. Integrating these features into a K-Nearest Neighbors algorithm showcased how to perform similarity searches, applying theoretical machine learning concepts in a practical scenario. Setting up a Flask web server provided hands-on experience in web development, particularly in handling file uploads and dynamically displaying content based on user interactions. This project highlighted the importance of seamlessly integrating multiple technologies—Python scripting, deep learning, and web development—to build a functional machine learning-driven web application. Overall, the experience was invaluable in understanding the workflow of creating end-to-end applications that combine web interfaces with backend machine learning processes.

