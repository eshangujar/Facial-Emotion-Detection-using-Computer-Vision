import os
import numpy as np
import cv2
from keras.models import load_model

from mtcnn.mtcnn import MTCNN 
detector = MTCNN()# MTCNN requires RGB format, but cv2 reads in BGR format.We will convert BGR to RGB later for it to work.


webcam = cv2.VideoCapture(0)

emotions_labels_dict={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

while True: 
    _, im = webcam.read()
    
    rgb_image = cv2.cvtColor(mini, cv2.COLOR_BGR2RGB) 
    faces = detector.detect_faces(rgb_image)

    for f in faces:

        face_img = im[y:y + h, x:x + w]
        
        resized = cv2.resize(face_img, (48,48))

        reshaped = np.reshape(resized, (1, 48,48, 3))
        reshaped/=255.0
        result = model.predict(reshaped) 
      

        label = np.argmax(result, axis=1)[0] 

        cv2.rectangle(im, (x, y), (x + w, y + h), (0,255,255), 2) 
        cv2.rectangle(im, (x, y - 40), (x + w, y), (0,255,255), -1) 
                                                                        
        cv2.putText(im, emotions_labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) 


    cv2.imshow('LIVE FACE DETECTION', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
