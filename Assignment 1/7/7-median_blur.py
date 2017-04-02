import cv2									#imports cv2 package
import numpy as np 							#imports numpy package
import matplotlib.pyplot as plt 			#imports matplotlib.pyplot package

def add_salt_and_pepper(gb, prob):
    '''Adds "Salt & Pepper" noise to an image.
    gb: should be one-channel image
    prob: probability (threshold) that controls level of noise'''
 
    rnd = np.random.rand(gb.shape[0], gb.shape[1])	#Create an array of size=resolution of image and
    											#populates it with random samples from a uniform distribution over [0, 1)
    noisy = gb.copy()						#creates a copy of the image
    noisy[rnd < prob] = 0					#sets all pixels with probability values less than threshold to black(0)
    noisy[rnd > 1 - prob] = 255				#sets all pixels with probability values greater than (1-threshold) to white(255) 
    return noisy 							#returns the image with salt & pepper noise added

img = cv2.imread('lena.jpg')				#reads the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#converts the image from BGR to Grayscale
noisy1 = add_salt_and_pepper(gray, 0.05)	#adds salt & pepper noise to the image
median1 = cv2.medianBlur(noisy1,5)			#smooths the noisy image using median blur of kernel size 5

pl1 = plt.subplot(131)						#creates a subplot in the figure
pl1.set_title('Original')					#sets a title to the image
pl1.imshow(gray, cmap='gray')				#plots the image using matplotlib, color map used is gray
pl1.axis("off")								#turns off the axes

pl1 = plt.subplot(132)
pl1.set_title('Noisy')
pl1.imshow(noisy1, cmap='gray')				#plots the image with salt and pepper noise added
pl1.axis("off")

pl2 = plt.subplot(133)
pl2.set_title('Median Blur')
pl2.imshow(median1, cmap='gray')			#plots the result of median blur
pl2.axis("off")

plt.show()									#displays the figure