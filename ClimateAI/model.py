from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

classes = {0: 'Cell', 1: 'Cell-Multi', 2: 'Cracking', 3: 'Diode', 4: 'Diode-Multi', 5: 'Hot-Spot', 6: 'Hot-Spot-Multi', 7: 'No-Anomaly', 8: 'Offline-Module', 9: 'Shadowing', 10: 'Soiling', 11: 'Vegetation'}

class CNN_Model:
    #    Class constructor, loads the dataset and loads the model
    #    if exists. If not, calls the _train_model method and 
    #    saves the model
    def __init__(self):
        self.model_fname_ = './saved_model/CNN_1'
        try:
            self.model = load_model(self.model_fname_)
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
    def predict_species(self, image):   
        data = img.imread(image)
        data = data.reshape(1, 40, 24, 1)
        probability = self.model.predict(data)
        prediction = np.argmax(probability)
        return classes.get(prediction[0])
