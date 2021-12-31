# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 03:00:33 2021

@author: Mahmoud
"""

import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_curve, auc



# dir = 'C:\\Users\\Mahmoud\\Documents\\archive\\trainingSet\\trainingSet'

# categories = ['one','two','0','3','4','5','6','7','8','9']
# data = []
# for category in categories :
#     path = os.path.join(dir,category)
#     label =categories.index(category)
    
#     for img in os.listdir(path):
#         imgpath = os.path.join(path,img)
#         pet_img =cv2.imread(imgpath,0)
#         try:
#             pet_img =cv2.resize(pet_img,(70,70))
#             image =np.array(pet_img).flatten()
#             data.append([image,label])
#         except Exception as e:
#             pass

# print(len(data))

# pick_in = open('data1.pickle','wb')
# pickle.dump(data,pick_in)
# pick_in.close()

pick_in = open('data1.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()


random.shuffle(data)
features = []
labels = []


for feature , label in data:
    features.append(feature)
    labels.append(label)
    
    
xtrain , xtest , ytrain , ytest = train_test_split(features , labels, test_size= 0.25)


model = SVC(C=1 , kernel='poly' , gamma = 'auto' )
model.fit(xtrain ,ytrain)

# pick_in = open('model.sav','wb')
# pickle.dump(model,pick_in)
# pick_in.close()

prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)

categories = ['one','two','0','3','4','5','6','7','8','9']

print ('Accuracy :' , accuracy)
print('prediction :', categories[prediction[0]])
print ('confuion matrix :', confusion_matrix(ytest,prediction))
plt.plot(confusion_matrix(ytest,prediction))
plt.show()
mypet= xtest[0].reshape(70,70)
plt.imshow(mypet, cmap='gray')
plt.show()


