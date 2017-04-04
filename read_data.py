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
a=0
for i in train_data:
    a=a+1
    print a
    # 
    # if a==10:
    #     break
    if len(train_data[i]['sequence'])<6:
        x = imread('./data/train/'+i)
        left = train_data[i]['left']
        top = train_data[i]['top']
        bottom = train_data[i]['bottom']
        right = train_data[i]['right']
        x = x[top:bottom, left:right]
        x = resize(x,(32,32,3))
        # x = x.reshape((1,32,32,3))
        # if a == 1:
        #     X = x
        # else:
        #     X=np.vstack((X,x))
        X.append(x.tolist())
        Y.append(len(train_data[i]['sequence']))

X = np.asarray(X)
Y = np.asarray(Y)

enc = OneHotEncoder(sparse=False)
Y =  enc.fit_transform(Y.reshape(-1,1))

print X.shape
print Y.shape

X.dump('cropped_images')
Y.dump('one_hot_sequence_len')
