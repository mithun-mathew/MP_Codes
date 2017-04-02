import cv2										#imports cv2 package
import numpy as np 								#imports numpy package
import matplotlib.pyplot as plt 				#imports matplotlib.pyplot package

img = cv2.imread('lena.jpg')					#reads the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	#converts the image from BGR to Grayscale
orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)		#converts the original image from BGR to RGB for display

ksize = 31										#sets size of Gabor kernel to be used
theta = (np.pi/16)*4							#sets the orientation of the normal to the parallel stripes of the Gabor function.
kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F) #creates the Gabor kernel
												'''
												4.0 is the standard deviation of the Gaussian function used
												10.0 is the wavelength of the sinusoidal factor of the Gabor filter
												0.5 is the spatial aspect ratio
												0 is the phase offset
												ktype indicates the type and range of values that each pixel in the Gabor kernel can hold
												'''
res = cv2.filter2D(gray, cv2.CV_8UC3, kern)		#applies the Gabor kernel to the image

pl1 = plt.subplot(121)							#creates a subplot in the figure
pl1.set_title('Original')						#sets a title to the image
pl1.imshow(orig, cmap='gray')					#plots the image using matplotlib, color map used is gray
pl1.axis("off")									#turns off the axes

pl2 = plt.subplot(122)
pl2.set_title('45 Degrees')
pl2.imshow(res, cmap='gray')					#plots the result with 45 degree strips detected
pl2.axis("off")

plt.show()										#displays the figure