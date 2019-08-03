

Face Detection using Haar Cascades
***************************************

::

    install opencv:
    pip instal opencv-python




In this We cover,

    * We will see the basics of face detection using Haar Feature-based Cascade Classifiers
    * We will extend the same for eye detection etc.
    
    
Basics
=========
    
Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

Here we will work with face detection. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. Then we need to extract features from it. For this, haar features shown in below image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under white rectangle from sum of pixels under black rectangle. 



Haar-cascade Detection in OpenCV
===================================

OpenCV comes with a trainer as well as detector. If you want to train your own classifier for any object like car, planes etc. you can use OpenCV to create one. Its full details are given here: `Cascade Classifier Training. <http://docs.opencv.org/doc/user_guide/ug_traincascade.html>`_

Here we will deal with detection. OpenCV already contains many pre-trained classifiers for face, eyes, smile etc. Those XML files are stored in ``opencv/data/haarcascades/`` folder. Let's create face and eye detector with OpenCV.

First we need to load the required XML classifiers. Then load our input image (or video) in grayscale mode.
::

    import numpy as np
    import cv2

    face_cs = cv2.CascadeClassifier('basepath/haarcascade_frontalface_default.xml')
    eye_cs = cv2.CascadeClassifier('basepath/haarcascade_eye.xml')

    img = cv2.imread('ravan.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


Now we find the faces in the image. If faces are found, it returns the positions of detected faces as Rect(x,y,w,h). Once we get these locations, we can create a ROI for the face and apply eye detection on this ROI (since eyes are always on the face !!! ).
::

    faces = face_cs.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cs.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


Result looks like below:

    .. image:: childface.jpg
        :alt: Face Detection
        :align: center


