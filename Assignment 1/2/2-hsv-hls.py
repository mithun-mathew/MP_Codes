import cv2									#imports cv2 package
import numpy as np 							#imports numpy package
import matplotlib.pyplot as plt 			#imports matplotlib.pyplot package

img = cv2.imread('rgbimage.jpg')			#reads the image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)	#converts the image from BGR to HSV
h,s,v = cv2.split(hsv)						#splits the HSV image to its components
orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)	#converts the original image from BGR to RGB for display

pl1 = plt.subplot(221)						#creates a subplot in the figure
pl1.set_title('original')					#sets a title to the image
pl1.imshow(orig, cmap='gray')				#plots the image using matplotlib, color map used is gray
pl1.axis("off")								#turns off the axes

pl2 = plt.subplot(222)
pl2.set_title('h')
pl2.imshow(h, cmap='hsv')					#plots the hue channel using color map 'hsv'
pl2.axis("off")

pl3 = plt.subplot(223)
pl3.set_title('s')
pl3.imshow(s, cmap='gray')					#plots the saturation channel
pl3.axis("off")

pl4 = plt.subplot(224)
pl4.imshow(v, cmap='gray')					#plots the value channel
pl4.set_title('v')
pl4.axis("off")

plt.show()									#displays the figure

hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)	#converts the image from BGR to HLS
h,l,s = cv2.split(hls)						#splits the HLS image to its components

pl1 = plt.subplot(221)						#creates a subplot in the figure
pl1.set_title('original')					#sets a title to the image
pl1.imshow(orig, cmap='gray')				#plots the image using matplotlib, color map used is gray
pl1.axis("off")								#turns off the axes

pl2 = plt.subplot(222)
pl2.set_title('h')
pl2.imshow(h, cmap='hsv')					#plots the hue channel using color map 'hsv'
pl2.axis("off")

pl3 = plt.subplot(223)
pl3.set_title('s')
pl3.imshow(s, cmap='gray')					#plots the saturation channel
pl3.axis("off")

pl4 = plt.subplot(224)
pl4.imshow(l, cmap='gray')					#plots the luminance channel
pl4.set_title('l')
pl4.axis("off")

plt.show()									#displays the figure