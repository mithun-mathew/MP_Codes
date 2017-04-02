import numpy as np
import cv2

file_name = 'dataset/c13.jpg'
img_orig = cv2.imread(file_name)
file_name = file_name.replace('.','/')
file_name = file_name.split('/')[1]

cv2.imshow(file_name+'_original_image', img_orig)
cv2.imwrite(file_name+'_original_image.jpg', img_orig)

img = cv2.medianBlur(img_orig,9)
res2 = cv2.pyrMeanShiftFiltering(img, sp=8, sr=16, maxLevel=1, termcrit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 5, 1))

canny = cv2.Canny(res2, 50, 240)
cv2.imshow(file_name+'_meanshift',res2)
cv2.imwrite(file_name+'_meanshift.jpg',res2)
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