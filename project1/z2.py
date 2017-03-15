import numpy as np
import cv2
import tensorflow as tf

data={
    "labels":np.zeros((10,10)),
    "images":np.zeros((10,784))
}

font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(10):

    img = np.zeros((400, 310), np.uint8)
    cv2.putText(img, str(i), (0, 370), font, 16, (255, 255, 255), 12)
    img=cv2.resize(img, (28, 28))
    cv2.imwrite("./images/"+str(i)+".png",img)
    temp=[0,0,0,0,0,0,0,0,0,0]
    temp[i]=1
    data["labels"][i]=temp
    data["images"][i]=img.flatten() / 255.0

#print(data)

x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#Train our model
iter = 1000
for i in range(iter):
    sess.run(train_step, feed_dict={x:data["images"],y_:data["labels"]})

#Evaluationg our model:
correct_prediction=tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
print("Accuracy: ", sess.run(accuracy, feed_dict={x:data["images"],y_:data["labels"]}))


# for i in range(10):
#     train_accuacy = accuracy.eval(feed_dict={x:data["images"],y_:data["labels"]})
#     print("step %d, training accuracy %g" % (i, train_accuacy))
#     train_step.run(feed_dict={x:data["images"],y_:data["labels"]})


my_classification = sess.run(tf.argmax(y, 1), feed_dict={x: [data["images"][5]]})
print('Neural Network predicted', my_classification[0], "for your digit")