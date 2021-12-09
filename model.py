from tensorflow.keras.models import load_model
import numpy as np

import tensorflow as tf

model = load_model('DSSL_fer')

class FacialExpressionModel():

    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self):
        self.loaded_model = load_model('DSSL_fer')

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

