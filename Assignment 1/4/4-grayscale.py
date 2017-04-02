import cv2									#imports cv2 package
import numpy as np 							#imports numpy package
import matplotlib.pyplot as plt 			#imports matplotlib.pyplot package

img = cv2.imread('rgbflag.jpg')				#reads the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#converts the image from BGR to Grayscale
orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)	#converts the original image from BGR to RGB for display

pl1 = plt.subplot(121)						#creates a subplot in the figure
pl1.set_title('Original')					#sets a title to the image
pl1.imshow(orig, cmap='gray')				#plots the image using matplotlib, color map used is gray
pl1.axis("off")								#turns off the axes

pl2 = plt.subplot(122)
pl2.set_title('Grayscale')
pl2.imshow(gray, cmap='gray')				#plots the grayscale image
pl2.axis("off")

plt.show()									#displays the figure