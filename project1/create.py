import numpy as np
import cv2

img=np.zeros((400,300),np.uint8)

font=cv2.FONT_HERSHEY_PLAIN

cv2.putText(img, "5", (-5, 370), font, 32, (255, 255, 255), 30)

kernel = np.ones((5,5),np.uint8)

#img = cv2.dilate(img,kernel,iterations = 1)
#img = cv2.GaussianBlur(img,(5,5),0)
img = cv2.erode(img,kernel,iterations = 1)

img=cv2.resize(img, (28, 28))

cv2.imwrite("create.png", img)