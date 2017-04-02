import cv2										#imports cv2 package
import numpy as np 								#imports numpy package
import matplotlib.pyplot as plt 				#imports matplotlib.pyplot package

img = cv2.imread('lena.jpg',0)					#reads the image as grayscale
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY) #applies binary thresholding to the image

kernx = np.matrix([[-1,0,1],[-1,0,1],[-1,0,1]])	#creates the Prewitt horizontal kernel
kerny = np.matrix([[-1,-1,-1],[0,0,0],[1,1,1]])	#creates the Prewitt vertical kernel
prewittx = cv2.filter2D(thresh1, -1, kernx)		#applies the Prewitt horizontal kernel to the image
prewitty = cv2.filter2D(thresh1, -1, kerny)		#applies the Prewitt vertical kernel to the image

pl1 = plt.subplot(221)							#creates a subplot in the figure
pl1.set_title('Original')						#sets a title to the image
pl1.imshow(img, cmap='gray')					#plots the image using matplotlib, color map used is gray
pl1.axis("off")									#turns off the axes

pl2 = plt.subplot(222)
pl2.set_title('Binary')
pl2.imshow(thresh1, cmap='gray')				#plot the binary image
pl2.axis("off")

pl2 = plt.subplot(223)
pl2.set_title('Prewitt X')
pl2.imshow(prewittx, cmap='gray')				#plots the image with Prewitt horizontal kernel applied
pl2.axis("off")

pl2 = plt.subplot(224)
pl2.set_title('Prewitt Y')
pl2.imshow(prewitty, cmap='gray')				#plots the image with Prewitt vertical kernel applied
pl2.axis("off")

plt.show()										#displays the figure

sobelx = cv2.Sobel(thresh1,-1,1,0,ksize=3)		#applies Sobel horizontal kernel of size 3 to the image
sobely = cv2.Sobel(thresh1,-1,0,1,ksize=3)		#applies Sobel vertical kernel of size 3 to the image

pl1 = plt.subplot(221)							#creates a subplot in the figure
pl1.set_title('Original')						#sets a title to the image
pl1.imshow(img, cmap='gray')					#plots the image using matplotlib, color map used is gray
pl1.axis("off")									#turns off the axes

pl2 = plt.subplot(222)
pl2.set_title('Binary')
pl2.imshow(thresh1, cmap='gray')				#plot the binary image
pl2.axis("off")

pl2 = plt.subplot(223)
pl2.set_title('Sobel X')
pl2.imshow(sobelx, cmap='gray')					#plots the image with Sobel horizontal kernel of size 3 applied
pl2.axis("off")

pl2 = plt.subplot(224)
pl2.set_title('Sobel Y')
pl2.imshow(sobely, cmap='gray')					#plots the image with Sobel vertical kernel of size 3 applied
pl2.axis("off")

plt.show()										#displays the figure