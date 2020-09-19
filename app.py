from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras import optimizers

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#Image utils
from tensorflow import keras
import cv2


# Define a flask app
app = Flask(__name__)

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Model saved with Keras model.save()
MODEL_PATH = 'models/model_ApperalDetection.h5'
BS = 256

# Load your trained model
loadedmodel = keras.models.load_model(MODEL_PATH)
adam = keras.optimizers.Adam(lr = 0.001)
loadedmodel.compile(optimizer=adam, loss='poisson', metrics=['accuracy'])
print('Model loaded and compiled with Adam optimizer. Start serving...')

# Check https://keras.io/applications/
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    image = cv2.imread(img_path)
    print("image shape", image.shape)
    image = cv2.resize(image, (28, 28))
    image_con = image.reshape((1, 28, 28, 3))
    predIdxs = loadedmodel.predict(image_con, batch_size=BS)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    kv = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker",
          8: "Bag", 9: "Ankle boot"}
    l = dict((k, v) for k, v in kv.items())
    prednames = l[predIdxs[0]]
    print(prednames)

    return prednames


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        model = keras.models.load_model("models/model")
        prednames = model_predict(file_path, model )
        return prednames
    return None


if __name__ == '__main__':
    app.run( debug = True, threaded = True)
    #In Cloud
    #app.run(host='0.0.0.0', debug = True, threaded = True)

    # Serve the app with gevent
    #http_server = WSGIServer(('0.0.0.0', 5000), app)
    #http_server.serve_forever()