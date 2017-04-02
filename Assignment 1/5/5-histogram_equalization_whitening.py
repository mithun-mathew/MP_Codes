import cv2									#imports cv2 package
import numpy as np 							#imports numpy package
import matplotlib.pyplot as plt 			#imports matplotlib.pyplot package

img = cv2.imread('lena.jpg')				#reads the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#converts the image from BGR to Grayscale

res1 = cv2.equalizeHist(gray)				#calculates the histogram of the array 'gray'
imgmean = np.mean(gray)						#calculates the mean of the array 'gray'
imgstd = np.std(gray)						#calculates the standard deviation of the array 'gray'
res2 = (gray-imgmean)/imgstd				#the whitening process

pl2 = plt.subplot(131)						#creates a subplot in the figure
pl2.set_title('Grayscale')					#sets a title to the image
pl2.imshow(gray, cmap='gray')				#plots the image using matplotlib, color map used is gray
pl2.axis("off")								#turns off the axes

pl3 = plt.subplot(132)
pl3.set_title('Histogram Equalization')
pl3.imshow(res1, cmap='gray')				#plots the result of Histogram Equalization
pl3.axis("off")

pl1 = plt.subplot(133)
pl1.set_title('Whitening')
pl1.imshow(res2, cmap='gray')				#plots the result of Whitening
pl1.axis("off")

plt.show()									#displays the figure