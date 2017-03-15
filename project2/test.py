import cv2
import numpy as np


im = cv2.imread('yan2.png')
im=cv2.resize(im,(301,120))
print(im.shape)

img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

kernel = np.ones((5,5),np.uint8)
img = cv2.dilate(img,kernel,iterations = 1)
img = cv2.GaussianBlur(img,(5,5),0)
ret1,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img = cv2.erode(img,kernel,iterations = 1) #膨胀

img=255-img

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


ret,thresh = cv2.threshold(img,127,255,0)

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


print(len(contours))
img = cv2.drawContours(im, contours, -1, (0,255,0), 3)

for i in range(len(contours)):
    print(cv2.contourArea(contours[i]))
    if cv2.contourArea(contours[i])>600:
        # epsilon = 0.1 * cv2.arcLength(contours[i], True)
        # approx = cv2.approxPolyDP(contours[i], epsilon, True)

        x, y, w, h = cv2.boundingRect(contours[i])
        imgtemp = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0,255), 2)

        rect = cv2.minAreaRect(contours[i])
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        cv2.imshow('image', imgtemp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()