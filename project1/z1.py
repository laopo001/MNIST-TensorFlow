import numpy as np
import cv2
import tensorflow as tf

data={
    "labels":np.zeros((60,10)),
    "images":np.zeros((60,784))
}

font = cv2.FONT_ITALIC
for i in range(10):
    img = np.zeros((400, 310), np.uint8)
    cv2.putText(img, str(i), (0, 370), font, 16, (255, 255, 255), 30)
    img=cv2.resize(img, (28, 28))
    cv2.imwrite("./images/"+str(i)+".png",img)
    temp=[0,0,0,0,0,0,0,0,0,0]
    temp[i]=1
    data["labels"][i]=temp
    data["images"][i]=img.flatten() / 255.0

    kernel = np.ones((5, 5), np.uint8)
    img2 = cv2.GaussianBlur(img, (5, 5), 0)
 #   img2 = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite("./images/" + str(i+10) + ".png", img2)
    data["labels"][i+10] = temp
    data["images"][i+10] = img2.flatten() / 255.0

    #img3 = cv2.GaussianBlur(img, (5, 5), 0)

    img3 = cv2.dilate(img, kernel, iterations=1)
    cv2.imwrite("./images/" + str(i + 20) + ".png", img3)
    data["labels"][i + 20] = temp
    data["images"][i + 20] = img3.flatten() / 255.0

font = cv2.FONT_HERSHEY_PLAIN
for i in range(10):
    img = np.zeros((400, 310), np.uint8)
    cv2.putText(img, str(i), (0, 370), font, 32, (255, 255, 255), 30)
    img=cv2.resize(img, (28, 28))
    cv2.imwrite("./images/"+str(i+30)+".png",img)
    temp=[0,0,0,0,0,0,0,0,0,0]
    temp[i]=1
    data["labels"][i+30]=temp
    data["images"][i+30]=img.flatten() / 255.0

    kernel = np.ones((5, 5), np.uint8)
    img2 = cv2.GaussianBlur(img, (5, 5), 0)
 #   img2 = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite("./images/" + str(i+40) + ".png", img2)
    data["labels"][i+40] = temp
    data["images"][i+40] = img2.flatten() / 255.0

    #img3 = cv2.GaussianBlur(img, (5, 5), 0)

    img3 = cv2.dilate(img, kernel, iterations=1)
    cv2.imwrite("./images/" + str(i + 50) + ".png", img3)
    data["labels"][i + 50] = temp
    data["images"][i + 50] = img3.flatten() / 255.0

#print(data)

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



print("开始")

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

# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


tf.summary.scalar('accuracy', accuracy)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.train.SummaryWriter('./log/mnist_logs', sess.graph)

sess.run(tf.initialize_all_variables())

for i in range(100):
    train_accuacy = accuracy.eval(feed_dict={x:data["images"],y_:data["labels"],keep_prob:1.0})
    print("step %d, training accuracy %g" % (i, train_accuacy))
    train_step.run(feed_dict={x:data["images"],y_:data["labels"],keep_prob:0.5})
    summary_str = merged_summary_op.eval(feed_dict={x:data["images"],y_:data["labels"],keep_prob:0.5})
    summary_writer.add_summary(summary_str, i)

print("test accuracy %g"%(accuracy.eval(feed_dict={x: data["images"], y_:data["labels"], keep_prob: 1.0})))

#my_classification = sess.run(tf.argmax(y_conv, 1), feed_dict={x: [data["images"][5]],keep_prob: 1.0})
#print('Neural Network predicted', my_classification[0], "for your digit")

saver = tf.train.Saver()
save_path = saver.save(sess, "./models/model.ckpt")
print("Model saved in file: ", save_path)

images = np.zeros((1,784))
gray = cv2.imread("my_digit.png", 0 )
gray = cv2.resize(255-gray, (28, 28))
# save the processed images
cv2.imwrite("my_grayscale_digit.png", gray)
flatten = gray.flatten() / 255.0
images[0] = flatten
my_classification = sess.run(tf.argmax(y_conv, 1), feed_dict={x: [images[0]],keep_prob: 1})
print('Neural Network predicted', my_classification[0], "for your digit")