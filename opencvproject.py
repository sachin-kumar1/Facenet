# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:52:13 2019

@author: sachi
"""
from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import win32com.client as wincl

PADDING = 40
ready_to_detect_identity = True
windows10_voice_interface = wincl.Dispatch("SAPI.SpVoice")
K.set_image_data_format('channels_first')
FRmodel = faceRecoModel(input_shape=(3, 96, 96))


def triplet_loss(y_true, y_pred, alpha = 0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,
               positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, 
               negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
   
    return loss

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)
FRmodel.summary()


def prepare_database():
    database = {}
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, FRmodel)
    return database

from collections import defaultdict
directory="C:/Users/sachi/Desktop/facenet/images/"
def labels_for_training_data(directory):

    database =[]
   

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")#Skipping files that startwith .
                continue

            id=os.path.basename(path)#fetching subdirectory names
            img_path=os.path.join(path,filename)#fetching image path
            test_img=img_path_to_encoding(img_path, FRmodel)
            #database[id] = tuple(test_img)
            
            #database.setdefault(id, []).append(test_img)
            database.append((id,test_img))
            print(database)
            print("img_path:",img_path)
            
            #print("id:",id)
    print(id)
    return(database)

for o in d.keys():
    for i in d[o]:
        print(o,i)

def who_is_it(image, database, model):
    
    encoding = img_to_encoding(image, model)
    
    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.
    for o in database.keys():
        for i in d[o]:
            
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
            dist = np.linalg.norm(i - encoding)
            print(dist)
            print('distance for %s is %s' %(o, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
            if dist < min_dist:
                min_dist = dist
                identity = o

    
    if min_dist > 0.50:
        return(0,'wwe')
    else:
        print(identity)
        return(1,identity)
        
        

def find_identity(frame, x1, y1, x2, y2):

    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    
    return who_is_it(part_image, database, FRmodel)


import cv2
cap = cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
x=labels_for_training_data(directory)
directory="C:/Users/sachi/Desktop/facenet/images/"
d=defaultdict(list)
for i, v in x:
    d[i].append(v)
database=d
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    
    for (x, y, w, h) in faces:

         x1 = x-PADDING
         y1 = y-PADDING
         x2 = x+w+PADDING
         y2 = y+h+PADDING
         
         identity,df = find_identity(frame, x1, y1, x2, y2)
         print(identity)
         

         if identity==1:
             name=df
             font=cv2.FONT_HERSHEY_SIMPLEX
             colors=(255,255,255)
             stroke=2
             cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = frame[y:y+h, x:x+w]
         #to get trian image
         #imgitem="train.jpg"
         #cv2.imwrite(imgitem,roi_color)
         image=roi_gray
         
         color = (255, 0, 0)
         stroke = 2
         end_cord_x = x + w
         end_cord_y = y + h
         cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    cv2.imshow('frame',frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


