import digit_predict
import numpy as np
import pickle
import gzip
from skimage.transform import resize

class SVHN(object):

    path = ""

    def __init__(self, data_dir):
        self.path = data_dir

    def train(self):
        digit_predict.train_model()


    def get_sequence(self, image):
        image = resize(image,(32,32,3))
        preds = digit_predict.get_predictions(image)
        preds = preds[1:preds[0]+2]
        return preds

    def save_model(self, **params):
        file_name = params['name']
        pickle.dump(self, gzip.open(file_name, 'wb'))


    @staticmethod
    def load_model(**params):
        digit_predict.load_model()
        file_name = params['name']
        return pickle.load(gzip.open(file_name, 'rb'))


if __name__ == "__main__":
        obj = SVHN('data/')
        # obj.train()
        obj.save_model(name="svhn.gz")
