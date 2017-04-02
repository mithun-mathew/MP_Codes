import cv2									#imports cv2 package
import numpy as np 							#imports numpy package
import matplotlib.pyplot as plt 			#imports matplotlib.pyplot package

img = cv2.imread('lena_noisy.jpg')			#reads the noisy image
orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)	#converts the original image from BGR to RGB for display

gaussian1 = cv2.GaussianBlur(img,(3,3),0)	#smooths the image using Gaussian kernel of size 3x3
gaussian2 = cv2.GaussianBlur(img,(7,7),0)	#smooths the image using Gaussian kernel of size 7x7
gaussian3 = cv2.GaussianBlur(img,(15,15),0)	#smooths the image using Gaussian kernel of size 15x15

pl1 = plt.subplot(221)						#creates a subplot in the figure
pl1.set_title('Original')					#sets a title to the image
pl1.imshow(orig, cmap='gray')				#plots the image using matplotlib, color map used is gray
pl1.axis("off")								#turns off the axes

pl2 = plt.subplot(222)
pl2.set_title('Gaussian Smoothing 1')
pl2.imshow(gaussian1, cmap='gray')			#plots the result of using 3x3 Gaussian kernel
pl2.axis("off")

pl2 = plt.subplot(223)
pl2.set_title('Gaussian Smoothing 2')
pl2.imshow(gaussian2, cmap='gray')			#plots the result of using 7x7 Gaussian kernel
pl2.axis("off")

pl2 = plt.subplot(224)
pl2.set_title('Gaussian Smoothing 3')
pl2.imshow(gaussian3, cmap='gray')			#plots the result of using 15x15 Gaussian kernel
pl2.axis("off")

plt.show()									#displays the figure