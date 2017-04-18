import digit_predict
import numpy as np

#digit_predict.train_model()
digit_predict.load_model()

X = np.load('dump/cropped_images')
Y = np.load('dump/one_hot_sequence_len')
Y_1 = np.load('dump/digit1')
Y_2 = np.load('dump/digit2')
Y_3 = np.load('dump/digit3')
Y_4 = np.load('dump/digit4')
Y_5 = np.load('dump/digit5')

validation_data_X = X[30000:]
validation_data_Y = Y[30000:]
validation_data_Y_1 = Y_1[30000:]
validation_data_Y_2 = Y_2[30000:]
validation_data_Y_3 = Y_3[30000:]
validation_data_Y_4 = Y_4[30000:]
validation_data_Y_5 = Y_5[30000:]

correct = 0

for i in range(len(validation_data_X)):
    print "Processing image", i, "of", len(validation_data_X)
    preds = digit_predict.get_predictions(validation_data_X[i])
    preds = preds[1:preds[0]+2]
    target = [np.argmax(validation_data_Y[i]), np.argmax(validation_data_Y_1[i]), np.argmax(validation_data_Y_2[i]), np.argmax(validation_data_Y_3[i]), np.argmax(validation_data_Y_4[i]), np.argmax(validation_data_Y_5[i])]
    target = target[1:target[0]+2]
    if preds == target:
        correct += 1
print "accuracy = ", correct/float(len(validation_data_X))
