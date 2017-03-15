import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
img = cv2.imread("yan4.png", 0)
img=cv2.resize(img,(301,120))

img = img[10:110,10:291]
#img = cv2.GaussianBlur(img,(9,9),0)

kernel = np.ones((5,5),np.uint8)

#img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

#ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)




img = cv2.dilate(img,kernel,iterations = 1)

img = cv2.GaussianBlur(img,(5,5),0)

ret1,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

img = cv2.erode(img,kernel,iterations = 1)


#img = cv2.Canny(img,100,200)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

width=64
heigth=96
img1=img[2:2+heigth,18:18+width]
img1=cv2.resize(255-img1,(28,28))

img2=img[2:2+heigth,77:77+width]
img2=cv2.resize(255-img2,(28,28))

img3=img[2:2+heigth,137:137+width]
img3=cv2.resize(255-img3,(28,28))

img4=img[2:2+heigth,198:198+width]
img4=cv2.resize(255-img4,(28,28))

list=[]
list.append(img1.flatten() / 255.0)
list.append(img2.flatten() / 255.0)
list.append(img3.flatten() / 255.0)
list.append(img4.flatten() / 255.0)
cv2.imwrite("./fen/digit1.png", img1)
cv2.imwrite("./fen/digit2.png", img2)
cv2.imwrite("./fen/digit3.png", img3)
cv2.imwrite("./fen/digit4.png", img4)
##########################################################
def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



sess = tf.InteractiveSession()
#tf.device("/gpu:0")
# paras
W_conv1 = weight_varible([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# conv layer-1
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv layer-2
W_conv2 = weight_varible([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# full connection
W_fc1 = weight_varible([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer: softmax
W_fc2 = weight_varible([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 10])






saver = tf.train.Saver()
saver.restore(sess, "./models/model.ckpt")

images = np.zeros((1,784))
#gray = cv2.imread("my_digit.png", 0 )




my_classification = sess.run(tf.argmax(y_conv, 1), feed_dict={x: list,keep_prob: 1})
print('Neural Network predicted', my_classification, "for your digit")



######################################################################
# ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# # Otsu's thresholding
# ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # Otsu's thresholding after Gaussian filtering
# #（5,5）为高斯核的大小，0 为标准差
# blur = cv2.GaussianBlur(img,(5,5),0)
# # 阈值一定要设为 0！
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # plot all the images and their histograms
# images = [img, 0, th1,
# img, 0, th2,
# blur, 0, th3]
# titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
# 'Original Noisy Image','Histogram',"Otsu's Thresholding",
# 'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
# # 这里使用了 pyplot 中画直方图的方法，plt.hist, 要注意的是它的参数是一维数组
# # 所以这里使用了（numpy）ravel 方法，将多维数组转换成一维，也可以使用 flatten 方法
# #ndarray.flat 1-D iterator over an array.
# #ndarray.flatten 1-D array copy of the elements of an array in row-major order.
# for i in range(3):
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()
