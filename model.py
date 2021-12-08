from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import numpy as np

import tensorflow as tf

model = load_model('DSSL_fer')

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)


class FacialExpressionModel():

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self):
        self.loaded_model = load_model('DSSL_fer')

    def predict_emotion(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]


[0.2, 0.3, 0.8, 0.3, 0.5, 0.7, 0.4]

