# import pickle
#
# objects = []
# with (open(r'C:\Users\jatoth.kumar\PycharmProjects\Tensorflow\sentiment_set.pickle', "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break
# for i in range(len(objects)):
#     print(i)
#     print(objects[i])
import tensorflow as tf
import numpy as np
from linked_list import here
train_vec,train_label,test_vec,test_label,inp = here()
print(train_vec[0],test_label[0],test_label[0],test_vec[0])
x = tf.placeholder("float")
y = tf.placeholder("float")
n_classes = 5
epochs = 100
# print(train_vec[0:1])
# print(train_vec[0])
# print(np.array([train_vec[0]]))
W = tf.Variable(tf.random_normal([len(train_vec[0]),n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))
def neuralnet(x):
    print(x.shape,W.shape)
    output = tf.matmul(x,W) + b
    return output
print(train_vec[0].shape)
batch_size = 12
saver = tf.train.Saver()
def neural_net(x):
    prediction = neuralnet(x)
    print("here in neural net")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for j in range(epochs):
            print(j)
            x1 = 0
            i = 0
            while i<len(train_vec):
                start = i
                end = i+batch_size
                train_x = np.array(train_vec[start:end])
                train_y = np.array(train_label[start:end])
                # print(train_x)
                # print(train_y)
                _, c = sess.run([optimizer, cost], feed_dict={x:train_x,y: train_y})
                x1 = x1+c
                i = i+batch_size
            print(x1)
            print("cost")
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # for i in range(len(test_label)):
        # saver.save(sess,'senti_model')
        print(len(test_vec),len(test_label),np.array(test_vec[0]).shape,np.array(test_label[0]).shape)
        k1 = (accuracy.eval({x: test_vec, y: test_label}))
        print(k1)
        saver.save(sess, 'checkpoints/my_test_model')
neural_net(x)

# if rnn is True:
    #go with multi to one Rnn
def neuralnet1(x):
    with tf.Session() as sess:
        saver.restore(sess, "checkpoints/my_test_model")
        pred_x = np.array(inp)
        print(pred_x)
        predictions = sess.run(neuralnet(x),feed_dict={x:pred_x})
        print(predictions)
        print(np.argmax(predictions))
neuralnet1(x)

