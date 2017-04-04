import sys
import numpy as np

train_data = dict()

with open('train.csv') as f:
    for j in f:
        i = j.split(',')
        if i[0]!='FileName':
            if i[0] not in train_data:
                train_data[i[0]] = dict()
                train_data[i[0]]['sequence']=[]
                train_data[i[0]]['left']=10000
                train_data[i[0]]['top']=10000
                train_data[i[0]]['right']=0
                train_data[i[0]]['bottom']=0
            if i[1]=='10':
                train_data[i[0]]['sequence'].append(0)
            else:
                train_data[i[0]]['sequence'].append(int(i[1]))
            left = max(0,int(i[2]))
            top = max(0,int(i[3]))
            right = int(i[4])+left
            bottom = int(i[5])+top

            if train_data[i[0]]['left'] > left:
                train_data[i[0]]['left'] = left

            if train_data[i[0]]['top'] > top:
                train_data[i[0]]['top'] = top

            if train_data[i[0]]['right'] < right:
                train_data[i[0]]['right'] = right

            if train_data[i[0]]['bottom'] < bottom:
                train_data[i[0]]['bottom'] = bottom

from skimage.viewer import ImageViewer
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder

X = []
Y = []
Y_1 = []
Y_2 = []
Y_3 = []
Y_4 = []
Y_5 = []

a=0
for i in train_data:
    a=a+1
    print a
    #
    # if a==10:
    #     break

    if len(train_data[i]['sequence'])<6:
        left = train_data[i]['left']
        top = train_data[i]['top']
        bottom = train_data[i]['bottom']
        right = train_data[i]['right']

        Y.append(len(train_data[i]['sequence']))

        if len(train_data[i]['sequence'])>=1:
            Y_1.append(train_data[i]['sequence'][0])
        else:
            Y_1.append(10)

        if len(train_data[i]['sequence'])>=2:
            Y_2.append(train_data[i]['sequence'][1])
        else:
            Y_2.append(10)

        if len(train_data[i]['sequence'])>=3:
            Y_3.append(train_data[i]['sequence'][2])
        else:
            Y_3.append(10)

        if len(train_data[i]['sequence'])>=4:
            Y_4.append(train_data[i]['sequence'][3])
        else:
            Y_4.append(10)

        if len(train_data[i]['sequence'])>=5:
            Y_5.append(train_data[i]['sequence'][4])
        else:
            Y_5.append(10)

        x = imread('./data/train/'+i)
        x = x[top:bottom, left:right]
        x = resize(x,(32,32,3))
        X.append(x.tolist())

X = np.asarray(X)
Y = np.asarray(Y)
Y_1 = np.asarray(Y_1)
Y_2 = np.asarray(Y_2)
Y_3 = np.asarray(Y_3)
Y_4 = np.asarray(Y_4)
Y_5 = np.asarray(Y_5)

enc1 = OneHotEncoder(sparse=False)
Y =  enc1.fit_transform(Y.reshape(-1,1))

enc = OneHotEncoder(n_values=11, sparse=False)
Y_1 =  enc.fit_transform(Y_1.reshape(-1,1))
Y_2 =  enc.fit_transform(Y_2.reshape(-1,1))
Y_3 =  enc.fit_transform(Y_3.reshape(-1,1))
Y_4 =  enc.fit_transform(Y_4.reshape(-1,1))
Y_5 =  enc.fit_transform(Y_5.reshape(-1,1))

print X.shape
print Y.shape
print Y_1.shape
print Y_2.shape
print Y_3.shape
print Y_4.shape
print Y_5.shape

X.dump('dump/cropped_images')
Y.dump('dump/one_hot_sequence_len')
Y_1.dump('dump/digit1')
Y_2.dump('dump/digit2')
Y_3.dump('dump/digit3')
Y_4.dump('dump/digit4')
Y_5.dump('dump/digit5')
