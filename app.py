import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Model file paths
model_dir = 'P:/Optipredict/models'
vgg_model_path = os.path.join(model_dir, 'vgg19_model_tested.keras')
resnet_model_path = os.path.join(model_dir, 'resnet50_model_tested.keras')
inception_model_path = os.path.join(model_dir, 'inceptionv3_model_tested.keras')

# Load models safely
try:
    vgg_model = load_model(vgg_model_path)
    resnet_model = load_model(resnet_model_path)
    inception_model = load_model(inception_model_path)
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

# Class labels
diseases = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'hypertension', 'macular', 'normal', 'pathological_myopia']

# Set allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing function
def preprocess_image(img_path, model_type='vgg19'):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Choose preprocessing function based on the model type
    if model_type == 'vgg19':
        img_array = vgg_preprocess(img_array)
    elif model_type == 'resnet50':
        img_array = resnet_preprocess(img_array)
    elif model_type == 'inceptionv3':
        img_array = inception_preprocess(img_array)

    return img_array

# Prediction function
def predict_disease(img_path, model_type='vgg19'):
    try:
        img_array = preprocess_image(img_path, model_type)
        if model_type == 'vgg19':
            predictions = vgg_model.predict(img_array)
        elif model_type == 'resnet50':
            predictions = resnet_model.predict(img_array)
        elif model_type == 'inceptionv3':
            predictions = inception_model.predict(img_array)

        # Get the prediction index (class with the highest probability)
        pred_idx = np.argmax(predictions, axis=1)[0]
        return diseases[pred_idx]
    except Exception as e:
        return f"Error in prediction: {e}"

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/doctors')
def doctors():
    return render_template('doctors.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blogs')
def blogs():
    return render_template('blogs.html')

@app.route('/book')
def book():
    return render_template('book.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/other')
def other():
    return render_template('other.html')

@app.route('/review')
def review():
    return render_template('review.html')

@app.route('/services')
def services():
    return render_template('services.html')

# Route for handling file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message="No file part.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message="No selected file.")
    
    if file and allowed_file(file.filename):
        # Ensure uploads directory exists
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Predict using all three models
        try:
            vgg19_prediction = predict_disease(file_path, 'vgg19')
            resnet50_prediction = predict_disease(file_path, 'resnet50')
            inceptionv3_prediction = predict_disease(file_path, 'inceptionv3')

            # Return predictions to the user
            return render_template('index.html', 
                                   vgg19_prediction=vgg19_prediction,
                                   resnet50_prediction=resnet50_prediction,
                                   inceptionv3_prediction=inceptionv3_prediction,
                                   filename=filename)
        except Exception as e:
            return render_template('index.html', message=f"Error during prediction: {e}")

    return render_template('index.html', message="Invalid file type. Only PNG, JPG, and JPEG are allowed.")

if __name__ == "__main__":
    app.run(debug=True)
