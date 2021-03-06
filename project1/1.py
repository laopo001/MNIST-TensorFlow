import tensorflow as tf
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
#使用numpy生成假数据phony data ，中共100个点。
x_data=np.float32(np.random.rand(2,100))
y_data=np.dot([0.1,0.2],x_data)+0.3

#构造一个线性模型
b=tf.Variable(tf.zeros([1]))
W=tf.Variable(tf.random_uniform([1,2],-1,1))
y=tf.matmul(W,x_data)+b

#最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#初始化变量
init = tf.initialize_all_variables()

#启动图
sess=tf.Session()
sess.run(init)

# 拟合平面
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
saver = tf.train.Saver()
save_path = saver.save(sess, "./models/model.ckpt")
print("Model saved in file: ", save_path)

sess.close()