from skeleton import SVHN
import numpy as np
import traceback
import math

class Evaluator(object):

    def __init__(self, data_dir):
        self.path = data_dir

    def load_images(self):
        self.X = np.load(self.path+'/cropped_images')
        self.Y = np.load(self.path+'/one_hot_sequence_len')
        self.Y_1 = np.load(self.path+'/digit1')
        self.Y_2 = np.load(self.path+'/digit2')
        self.Y_3 = np.load(self.path+'/digit3')
        self.Y_4 = np.load(self.path+'/digit4')
        self.Y_5 = np.load(self.path+'/digit5')

    def evaluate(self, model):
        correct = 0

        for i in range(len(self.X)):
            print "Processing image", i, "of", len(self.X)
            target = [np.argmax(self.Y[i]), np.argmax(self.Y_1[i]), np.argmax(self.Y_2[i]), np.argmax(self.Y_3[i]), np.argmax(self.Y_4[i]), np.argmax(self.Y_5[i])]
            target = target[1:target[0]+2]
            try:
                preds = model.get_sequence(np.asarray(self.X[i], dtype=np.uint8))
                if preds == target:
                    correct += 1
            except Exception as e:
                print traceback.print_exc()

        accuracy = correct/float(len(self.X))
        score = 0
        if accuracy <0.5:
            score = 15.0 * accuracy
        else:
            score = 7.5*(math.exp(1.386*(accuracy-0.5)))

        print "Accuracy =", accuracy*100
        print "Marks =", score

if __name__ == "__main__":

        evaluator = Evaluator("dump_test")
        evaluator.load_images()
        obj = SVHN.load_model(name='svhn.gz')
        evaluator.evaluate(obj)
