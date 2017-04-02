import cv2										#imports cv2 package
import numpy as np 								#imports numpy package
import matplotlib.pyplot as plt 				#imports matplotlib.pyplot package

img = cv2.imread('lena.jpg',0)					#reads the image as grayscale
laplacian = cv2.Laplacian(img,-1)				#applies Laplacian filtering to the image
img = img.astype(np.int16)						#datatype conversion to int16
out = img - laplacian 							#sharpens the image
out = out.clip(min=0)							#limits the values in the array to a minimum of 0

pl1 = plt.subplot(131)							#creates a subplot in the figure
pl1.set_title('Grayscale')						#sets a title to the image
pl1.imshow(img, cmap='gray')					#plots the image using matplotlib, color map used is gray
pl1.axis("off")									#turns off the axes

pl1 = plt.subplot(132)
pl1.set_title('Laplacian')
pl1.imshow(laplacian, cmap='gray')				#plots the result of Laplacian filtering
pl1.axis("off")

pl2 = plt.subplot(133)
pl2.set_title('Sharpened')
pl2.imshow(out, cmap='gray')					#plots the result of sharpening using Laplacian
pl2.axis("off")

plt.show()										#displays the figure