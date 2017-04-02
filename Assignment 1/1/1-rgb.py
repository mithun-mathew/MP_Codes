import cv2									#imports cv2 package
import numpy as np 							#imports numpy package
import matplotlib.pyplot as plt 			#imports matplotlib.pyplot package

img = cv2.imread('rgbflag.jpg') 			#reads the image
blue, green, red = cv2.split(img)			#splits the image into component channels
orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)	#converts the image from BGR to RGB, which is used by matplotlib, to display the result

pl1 = plt.subplot(221)						#creates a subplot in the figure
pl1.set_title('Original')					#sets a title to the image
pl1.imshow(orig, cmap='gray')				#plots the image using matplotlib, color map used is gray
pl1.axis("off")								#turns off the axes

pl2 = plt.subplot(222)
pl2.set_title('Blue')
pl2.imshow(blue, cmap='gray')				#plots the blue channel
pl2.axis("off")

pl3 = plt.subplot(223)
pl3.set_title('Green')
pl3.imshow(green, cmap='gray')				#plots the green channel
pl3.axis("off")

pl4 = plt.subplot(224)
pl4.imshow(red, cmap='gray')				#plots the red channel
pl4.set_title('Red')
pl4.axis("off")

plt.show()									#displays the figure