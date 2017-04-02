import numpy as np
import cv2

file_name = 'dataset/c13.jpg'
img_orig = cv2.imread(file_name)
file_name = file_name.replace('.','/')
file_name = file_name.split('/')[1]

cv2.imshow(file_name+'_original_image', img_orig)
cv2.imwrite(file_name+'_original_image.jpg', img_orig)

img = cv2.medianBlur(img_orig,9)
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

canny = cv2.Canny(res2, 50, 240)
cv2.imshow(file_name+'_kmeans',res2)
cv2.imwrite(file_name+'_kmeans.jpg',res2)
cv2.imshow(file_name+'_canny', canny)
cv2.imwrite(file_name+'_canny.jpg', canny)

for i in range(canny.shape[0]):
  for j in range(canny.shape[1]):
    if canny[i][j] != 0:
      img_orig[i][j][0] = 0
      img_orig[i][j][1] = 255
      img_orig[i][j][2] = 0

cv2.imshow(file_name+'_final_result', img_orig)
cv2.imwrite(file_name+'_final_result.jpg', img_orig)
cv2.waitKey(0)
cv2.destroyAllWindows()