import os
import re
import cv2
import numpy as np
from keras.models import load_model


def predict(X):
    # Read the images
    image_id = []
    for filename in X:
        if re.findall(r"\.png", filename):
            img = cv2.imread(filename)
            image_id += [(img, filename.strip(r"\.png"))]
    res = np.zeros([len(image_id), 32, 32, 3])
    # Resize the images
    for i in range(len(image_id)):
        res[i] = cv2.resize(image_id[i][0], (32, 32), interpolation=cv2.INTER_CUBIC)
    # Reshape the data
    x_total = res.reshape(len(res), -1)
    # Scale the data
    x_test = x_total / 255
    # Load the model
    model1 = load_model('mlp_yongchaoqiao91.hdf5')
    model2 = load_model('mlp_yongchaoqiao92.hdf5')

    # Get the predict
    y_pred = np.argmax((model1.predict(x_test) + model2.predict(x_test)) / 2, axis=1)

    return y_pred, model1, model2
