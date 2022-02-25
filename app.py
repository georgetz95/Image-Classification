import numpy as np
from scipy import stats
from scipy.special import softmax
from flask import Flask, render_template, request
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import pickle
import os

app = Flask(__name__)

# Create paths
base_path = os.getcwd()
upload_path = os.path.join(base_path, 'static/upload/')
models_path = os.path.join(base_path, 'static/models/')
scaler_path = os.path.join(models_path, 'scaler.pkl')
sgd_path = os.path.join(models_path, 'model_final.pkl')
print(scaler_path)
# Load models
scaler = pickle.load(open(scaler_path, 'rb'))
sgd = pickle.load(open(sgd_path, 'rb'))

@app.errorhandler(404)
def error_404(error):
    message = 'Error 404: Page Not Found. Please reload the main page.'
    return render_template('error.html', message=message)

@app.errorhandler(405)
def error_405(error):
    message = 'Error 405: Method Not Found. Please reload the main page.'
    render_template('error.html', message=message)

@app.errorhandler(500)
def error_500(error):
    message = 'Error 500: Internal Server Error. Please reload the main page.'
    render_template('error.html', message=message)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        extension = filename.split('.')[-1]
        print(f"File name: {filename}")
        print(f"File extension: {extension}")
        if extension.lower() in ['png', 'jpg', 'jpeg']:
            image_path = os.path.join(upload_path, filename)
            upload_file.save(image_path)
            print(f"File uploaded successfully.")

            results = pipeline_model(image_path, scaler, sgd)
            height = get_height(image_path)
            print(results)
            return render_template('upload.html', fileupload=True, data=results, image_filename=filename, height=height)
            
        else:
            print('Wrong file extension.')
            return render_template('upload.html', fileupload=False, extension=True)

    else:
        return render_template('upload.html', fileupload=False, extension=False)

@app.route('/about')
def about():
    return render_template('about.html')

def get_height(path):
    image = imread(path)
    if len(image.shape) == 2:
        h, w = image.shape
    elif len(image.shape) == 3:
        h, w, _ = image.shape
    aspect = h / w
    given_width=250
    height = given_width * aspect
    return height


def pipeline_model(path, scaler, model):
    # Reading Image
    image = imread(path)
    # Resizing Image
    image_transformed = resize(image, (80,80), preserve_range=True).astype(np.uint8)
    # Grayscale Image
    if len(image.shape) == 3:
        if image.shape[-1] == 3:
            image_gray = rgb2gray(image_transformed)
        elif image.shape[-1] == 4:
            image_gray = rgb2gray(rgba2rgb(image_transformed))
    elif len(image.shape) == 2:
        image_gray = image_transformed
        
    # Hog Features
    feature_vector = hog(image_gray, orientations=8, pixels_per_cell=(8,8), cells_per_block=(2,2))
    # Scaling Image
    scalex = scaler.transform(feature_vector.reshape(1, -1))
    # Predicting Class
    result = model.predict(scalex)
    # Decision Function
    decision_values = model.decision_function(scalex).flatten()
    classes = model.classes_
    # Z-Scores
    z_scores = stats.zscore(decision_values)
    # Probabilities
    prob_values = softmax(z_scores)
    # Top 5 Results
    top_5_ind = prob_values.argsort()[::-1][:5]
    top_5_classes = classes[top_5_ind]
    top_5_probs = prob_values[top_5_ind]
    top_dict = {key: np.round(value, 2) for key, value in zip(top_5_classes, top_5_probs)}
    
    return top_dict
    
    
    
    
    
    



if __name__ == '__main__':
    app.run(debug=True)