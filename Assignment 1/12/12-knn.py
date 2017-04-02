import cv2                                  #imports cv2 package
import numpy as np                          #imports numpy package
from imutils import paths                   #imports paths module
from sklearn.metrics import accuracy_score  #imports accuracy_score module
import os                                   #imports os module
from matplotlib import pyplot as plt        #imports matplotlib.pyplot package

classes = {1: 'Portrait',
           2: 'Landscape',
           3: 'Night'}                      #creates a python dictionary for the various classes
train_features = []                         #creates a list to store the feature vectors for the images
labels = []                                 #creates a list to store the classes associated with the images

imagePaths = list(paths.list_images("dataset")) #specifies the location of the set of training images
for imagePath in enumerate(imagePaths):         #loop over the images in the location
    image = cv2.imread(imagePath[1])            #read each image
    label = imagePath[1].split(os.path.sep)[-1].split(".")[0]   #get the class name of each image in the training set from its filename
    if label == 'portrait':                     #according to the class of the image, set a corresponding label number
        label = 1
    elif label == 'landscape':
        label = 2
    elif label == 'night':
        label = 3
    labels.append(label)                    #appends the class name to the list
    train_features.append(cv2.resize(image, (640, 480)).flatten()) #appends the feature vector to the list

train_features = np.array(train_features).astype(np.float32)    #convert the feature vector list to a numpy array and convert the datatype to float32
labels = np.array(labels).astype(np.float32)                    #convert the class name list to a numpy array and convert the datatype to float32
knn = cv2.ml.KNearest_create()                                  #implements the K-Nearest Neighbors model
knn.train(train_features, cv2.ml.ROW_SAMPLE, labels)            #trains the model

expected_label = []                         #creates a list to store the expected class names of the test images
output_label = []                           #creates a list to store the predicted class names of the test images
testPaths = list(paths.list_images("test")) #specifies the location of the set of test images
for testPath in enumerate(testPaths):       #loop over the images in the location
    testimage = cv2.imread(testPath[1])     #read each image
    label = testPath[1].split(os.path.sep)[-1].split(".")[0] #get the class name of each image in the test image set from its filename
    if label == 'portrait':                 #according to the class of the image, set a corresponding label number
        label = 1
    elif label == 'landscape':
        label = 2
    elif label == 'night':
        label = 3
    expected_label.append(label)            #append the class name to the expected class names list

    test_data = []
    test_data.append(cv2.resize(testimage, (640, 480)).flatten())   #get feature vector of the test image
    test_data = np.array(test_data).astype(np.float32)              #convert the feature vector list to a numpy array and convert the datatype to float32
    ret, result, neighbour, distance = knn.findNearest(test_data, 3)#finds the neighbors and predicts responses for test images
    print(testPath[1], "-", classes[result[0][0]])                  #prints the predicted class name to the terminal
    output_label.append(result[0][0])                               #append the predicted class name to the list of predicted class names
    testimage = cv2.cvtColor(testimage, cv2.COLOR_BGR2RGB)          #converts the image from BGR to RGB for display
    plt.subplot(111),plt.imshow(testimage,cmap = 'gray')            #plots the test image
    plt.title(classes[result[0][0]]), plt.xticks([]), plt.yticks([])#shows the predicted class name
    plt.show()                                                      #displays the figure

print("\nAccuracy - ",accuracy_score(expected_label, output_label)*100, "%")    #displays the accuracy level of the predictions to the terminal