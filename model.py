import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import pandas as pd

class CNN_Model:
    #    Class constructor, loads the dataset and loads the model
    #    if exists. If not, calls the _train_model method and 
    #    saves the model
    def __init__(self):
        self.model_fname_ = 'saved_model/CNN_1'
        try:
            self.model = tf.keras.models.load_model(self.model_fname_)
        except Exception as _:
            '''Unfinished:
            self.model = self._train_model()
            self.model.save(self.model_fname_)'''
        

    # 4. Perform model training using the RandomForest classifier
    def _train_model(self):
        '''currently in progress'''
        #return model


    # 5. Make a prediction based on the user-entered data
    #    Returns the predicted species with its respective probability
    def predict_species(self, data):   
        probability = self.model.predict(data)
        prediction = np.argmax(probability)
        return prediction[0], probability
