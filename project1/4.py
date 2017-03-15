import tensorflow as tf
input1=tf.constant(3)
input2=tf.constant(2)
input3=tf.constant(5)
intermed=tf.add(input2,input3)
mul=tf.mul(input1,intermed)

with tf.Session() as sess:
    result=sess.run([mul,intermed])
    print(result)