import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
img = cv2.imread("yan3.png", 0)
img=cv2.resize(img,(301,120))

#img = img[10:110,10:291]
#img = cv2.GaussianBlur(img,(9,9),0)

kernel = np.ones((5,5),np.uint8)

#img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

#ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)




img = cv2.dilate(img,kernel,iterations = 1)

img = cv2.GaussianBlur(img,(5,5),0)

ret1,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

img = cv2.erode(img,kernel,iterations = 1) #膨胀


#ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0, 0, 255), 3)


#img = cv2.Canny(img,100,200)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()