import numpy as np
import cv2

img1 = cv2.imread('Panorama/secondPic1.jpg')
img2 = cv2.imread('Panorama/secondPic2.jpg')

cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)

surf = cv2.xfeatures2d.SURF_create(400)
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1, des2, k=2)
goodMatches = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        goodMatches.append(m)
if len(goodMatches) > 4:
    img1_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatches])
    img2_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatches])
    (M, mask) = cv2.findHomography(img2_pts, img1_pts, cv2.RANSAC, 5.0)
    result = cv2.warpPerspective(img2, M, (img1.shape[1] + img2.shape[1], img2.shape[0]))
    result[0:img1.shape[0], 0:img1.shape[1]] = img1
    cv2.imwrite('Panorama.jpg', result)
    cv2.waitKey(0)
