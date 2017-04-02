import cv2										#imports cv2 package
import numpy as np 								#imports numpy package
import matplotlib.pyplot as plt 				#imports matplotlib.pyplot package

img_noblur = cv2.imread('road8.jpg')			#reads the image
imgnew = img_noblur.copy()						#creates a copy of the image
img_noblur_grey = cv2.cvtColor(img_noblur, cv2.COLOR_BGR2GRAY)	#converts the image from BGR to Grayscale
img = cv2.GaussianBlur(img_noblur_grey,(5,5),0)	#applies a Gaussian Blur to the image for smoothing

sobelx = cv2.Sobel(img,-1,1,0,ksize=3)			#applies Sobel horizontal kernel of size 3 to the image
sobelx[sobelx<100] = 0							#discards low intensity pixels

lines = cv2.HoughLinesP(sobelx,1,np.pi/180,100)	#use HoughLinesP to detect lines in the image to which Sobel horizontal kernel was applied
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(imgnew,(x1,y1),(x2,y2),(0,255,0),5)		#draws the detected lines on the image

imgnew = cv2.cvtColor(imgnew, cv2.COLOR_BGR2RGB)			#converts the image from BGR to RGB
img_noblur = cv2.cvtColor(img_noblur, cv2.COLOR_BGR2RGB)	#converts the original image from BGR to RGB for display

plt.subplot(131),plt.imshow(img_noblur,cmap = 'gray')		#plots the original image
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(sobelx,cmap = 'gray')			#plots the result of applying Sobel horizontal kernel to the image
plt.title('Sobel'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(imgnew,cmap = 'gray')			#plots the result with the road markers detected
plt.title('Output'), plt.xticks([]), plt.yticks([])

plt.show()											#displays the figure