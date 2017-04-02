import cv2
import numpy as np
from imutils import paths  
import os 
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

classes = {1: 'Bike',
           2: 'Horse'} 
labels = []
dictionarySize = 12

BOW = cv2.BOWKMeansTrainer(dictionarySize)
sift = cv2.xfeatures2d.SIFT_create()

imagePaths = list(paths.list_images("train"))
for image in imagePaths:
	label = image.split('/')[1]
	if label == 'Bike':
		labels.append(1)
	elif label == 'Horse':
		labels.append(2)

	image = image.replace("\\","")
	img = cv2.imread(image,0)
	kp, des = sift.detectAndCompute(img,None)
	BOW.add(des)

dictionary = BOW.cluster()
print("Created Bag of Words")

sift2 = cv2.xfeatures2d.SIFT_create()
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)

desc = []
for image in imagePaths:
	image = image.replace("\\","")
	img = cv2.imread(image,0)	
	desc.extend(bowDiction.compute(img, sift.detect(img)))

desc = np.array(desc).astype(np.float32)
labels = np.array(labels).astype(np.float32) 
knn = cv2.ml.KNearest_create()
knn.train(desc, cv2.ml.ROW_SAMPLE, labels)
print("Created KNN model")

expected_label = []
output_label = [] 
testPath = list(paths.list_images("test"))
print('Actual Class','\t-\t','Predicted Class')
for image in testPath:
	label = image.split('/')[1]
	image = image.replace("\\","")
	img = cv2.imread(image,0)

	feature = bowDiction.compute(img, sift.detect(img))
	feature = np.array(feature).astype(np.float32)
	ret, result, neighbour, distance = knn.findNearest(feature, 3)
	print(label, "\t\t-\t", classes[result[0][0]])
	expected_label.append(label)
	output_label.append(classes[result[0][0]])

	img = cv2.imread(image)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	plt.subplot(111),plt.imshow(img,cmap = 'gray')
	plt.title(classes[result[0][0]]), plt.xticks([]), plt.yticks([])
	plt.show()

print("\nAccuracy - ",accuracy_score(expected_label, output_label)*100, "%")