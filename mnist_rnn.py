import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
mnist=input_data.read_data_sets("MNIST-data/",one_hot=True)
# Architecture
# Inp(128,28,28)->unstack(128,28) of 28 timesteps So we have 28 lstm blocks
# Lstm Output shape (128,128) (batch_size,num_units)
# Weights of shape (128,10)
# bias of shape (None,10)
# prediction is of shape (128,10)
# This we will be taking only for the last output layers of 128 hidden layers in the LSTM of one block
#
inpsize = 28
batchsize = 128
timesteps = 28
learningrate = 0.001
n_classes = 10
n_units = 128

weights = tf.Variable(tf.random_normal([n_units,n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

x = tf.placeholder("float",[None,timesteps,inpsize])
y = tf.placeholder("float",[None,n_classes])
input = tf.unstack(x,timesteps,1)
print("input size")
print(input[-1])
lstm = rnn.BasicLSTMCell(num_units=n_units,forget_bias=1.0)
outputs,_ = rnn.static_rnn(cell=lstm,inputs=input,dtype="float32")
print("len of outputs")
print(len(outputs))
print(outputs[-1])
prediction = tf.matmul(outputs[-1],weights) + bias
print("prediction")
print(prediction.shape)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningrate).minimize(loss)

correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter = 1
    while(iter < 80):
        batch_x,batch_y = mnist.train.next_batch(batch_size=batchsize)
        batch_x = batch_x.reshape((batchsize,timesteps,inpsize))
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if iter % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            print("For iter ", iter)
            print("Accuracy ", acc)
            print("Loss ", los)
            print('__________________')
        iter= iter+1

    test_data = mnist.test.images[:128].reshape((-1,timesteps,inpsize))
    test_lable = mnist.test.labels[:128]
    print(sess.run(accuracy,feed_dict={x:test_data,y:test_lable}))
