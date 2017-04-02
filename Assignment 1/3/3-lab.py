import cv2									#imports cv2 package
import numpy as np 							#imports numpy package
import matplotlib.pyplot as plt 			#imports matplotlib.pyplot package

img = cv2.imread('rgbflag.jpg')				#reads the image
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)	#converts the image from BGR to L*a*b*
orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)	#converts the original image from BGR to RGB for display

pl1 = plt.subplot(121)						#creates a subplot in the figure
pl1.set_title('Original')					#sets a title to the image
pl1.imshow(orig, cmap='gray')				#plots the image using matplotlib, color map used is gray
pl1.axis("off")								#turns off the axes

pl4 = plt.subplot(122)
pl4.imshow(lab, cmap='gray')
pl4.set_title('lab')						#plots the L*a*b* image
pl4.axis("off")

plt.show()									#displays the figure