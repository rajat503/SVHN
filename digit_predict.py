import numpy as np
import tensorflow as tf
import random

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 5])
y_1 = tf.placeholder(tf.float32, shape=[None, 11])
y_2 = tf.placeholder(tf.float32, shape=[None, 11])
y_3 = tf.placeholder(tf.float32, shape=[None, 11])
y_4 = tf.placeholder(tf.float32, shape=[None, 11])
y_5 = tf.placeholder(tf.float32, shape=[None, 11])


lr = tf.placeholder(tf.float32)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([3, 3, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 32, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([3, 3, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

W_conv4 = weight_variable([3, 3, 64, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

h_pool4_flat = tf.reshape(h_pool4, [-1, 4*4*128])

W_fc1 = weight_variable([4 * 4 * 128, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)

W_fc3 = weight_variable([1024, 1024])
b_fc3 = bias_variable([1024])
h_fc3 = tf.nn.relu(tf.matmul(h_fc1, W_fc3) + b_fc3)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

W_fc4 = weight_variable([1024, 5])
b_fc4 = bias_variable([5])
y_conv=tf.nn.softmax(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

cross_entropy_s = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), reduction_indices=[1]))

W_fc3_1 = weight_variable([1024, 1024])
b_fc3_1 = bias_variable([1024])
h_fc3_1 = tf.nn.relu(tf.matmul(h_fc1, W_fc3_1) + b_fc3_1)
h_fc3_drop_1 = tf.nn.dropout(h_fc3_1, keep_prob)

W_fc4_1 = weight_variable([1024, 11])
b_fc4_1 = bias_variable([11])
y_conv_1=tf.nn.softmax(tf.matmul(h_fc3_drop_1, W_fc4_1) + b_fc4_1)

cross_entropy_1 = tf.reduce_mean(-tf.reduce_sum(y_1*tf.log(tf.clip_by_value(y_conv_1,1e-10,1.0)), reduction_indices=[1]))

W_fc3_2 = weight_variable([1024, 1024])
b_fc3_2 = bias_variable([1024])
h_fc3_2 = tf.nn.relu(tf.matmul(h_fc1, W_fc3_2) + b_fc3_2)
h_fc3_drop_2 = tf.nn.dropout(h_fc3_2, keep_prob)

W_fc4_2 = weight_variable([1024, 11])
b_fc4_2 = bias_variable([11])
y_conv_2=tf.nn.softmax(tf.matmul(h_fc3_drop_2, W_fc4_2) + b_fc4_2)

cross_entropy_2 = tf.reduce_mean(-tf.reduce_sum(y_2*tf.log(tf.clip_by_value(y_conv_2,1e-10,1.0)), reduction_indices=[1]))

W_fc3_3 = weight_variable([1024, 1024])
b_fc3_3 = bias_variable([1024])
h_fc3_3 = tf.nn.relu(tf.matmul(h_fc1, W_fc3_3) + b_fc3_3)
h_fc3_drop_3 = tf.nn.dropout(h_fc3_3, keep_prob)

W_fc4_3 = weight_variable([1024, 11])
b_fc4_3 = bias_variable([11])
y_conv_3=tf.nn.softmax(tf.matmul(h_fc3_drop_3, W_fc4_3) + b_fc4_3)

cross_entropy_3 = tf.reduce_mean(-tf.reduce_sum(y_3*tf.log(tf.clip_by_value(y_conv_3,1e-10,1.0)), reduction_indices=[1]))

W_fc3_4 = weight_variable([1024, 1024])
b_fc3_4 = bias_variable([1024])
h_fc3_4 = tf.nn.relu(tf.matmul(h_fc1, W_fc3_4) + b_fc3_4)
h_fc3_drop_4 = tf.nn.dropout(h_fc3_4, keep_prob)

W_fc4_4 = weight_variable([1024, 11])
b_fc4_4 = bias_variable([11])
y_conv_4=tf.nn.softmax(tf.matmul(h_fc3_drop_4, W_fc4_4) + b_fc4_4)

cross_entropy_4 = tf.reduce_mean(-tf.reduce_sum(y_4*tf.log(tf.clip_by_value(y_conv_4,1e-10,1.0)), reduction_indices=[1]))

W_fc3_5 = weight_variable([1024, 1024])
b_fc3_5 = bias_variable([1024])
h_fc3_5 = tf.nn.relu(tf.matmul(h_fc1, W_fc3_5) + b_fc3_5)
h_fc3_drop_5 = tf.nn.dropout(h_fc3_5, keep_prob)

W_fc4_5 = weight_variable([1024, 11])
b_fc4_5 = bias_variable([11])
y_conv_5=tf.nn.softmax(tf.matmul(h_fc3_drop_5, W_fc4_5) + b_fc4_5)

cross_entropy_5 = tf.reduce_mean(-tf.reduce_sum(y_5*tf.log(tf.clip_by_value(y_conv_5,1e-10,1.0)), reduction_indices=[1]))

cross_entropy  = cross_entropy_1 + cross_entropy_2 + cross_entropy_3 + cross_entropy_4 + cross_entropy_5 + cross_entropy_s

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

arg_y = tf.argmax(y_conv,1)
arg_y_1 = tf.argmax(y_conv_1,1)
arg_y_2 = tf.argmax(y_conv_2,1)
arg_y_3 = tf.argmax(y_conv_3,1)
arg_y_4 = tf.argmax(y_conv_4,1)
arg_y_5 = tf.argmax(y_conv_5,1)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

correct_prediction_1 = tf.equal(tf.argmax(y_conv_1,1), tf.argmax(y_1,1))
accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction_1, tf.float32))

correct_prediction_2 = tf.equal(tf.argmax(y_conv_2,1), tf.argmax(y_2,1))
accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))

correct_prediction_3 = tf.equal(tf.argmax(y_conv_3,1), tf.argmax(y_3,1))
accuracy_3 = tf.reduce_mean(tf.cast(correct_prediction_3, tf.float32))

correct_prediction_4 = tf.equal(tf.argmax(y_conv_4,1), tf.argmax(y_4,1))
accuracy_4 = tf.reduce_mean(tf.cast(correct_prediction_4, tf.float32))

correct_prediction_5 = tf.equal(tf.argmax(y_conv_5,1), tf.argmax(y_5,1))
accuracy_5 = tf.reduce_mean(tf.cast(correct_prediction_5, tf.float32))

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())


def train_model():
    X = np.load('dump/cropped_images')
    Y = np.load('dump/one_hot_sequence_len')
    Y_1 = np.load('dump/digit1')
    Y_2 = np.load('dump/digit2')
    Y_3 = np.load('dump/digit3')
    Y_4 = np.load('dump/digit4')
    Y_5 = np.load('dump/digit5')


    train_data_X = X[:30000]
    train_data_Y = Y[:30000]
    train_data_Y_1 = Y_1[:30000]
    train_data_Y_2 = Y_2[:30000]
    train_data_Y_3 = Y_3[:30000]
    train_data_Y_4 = Y_4[:30000]
    train_data_Y_5 = Y_5[:30000]


    validation_data_X = X[30000:]
    validation_data_Y = Y[30000:]
    validation_data_Y_1 = Y_1[30000:]
    validation_data_Y_2 = Y_2[30000:]
    validation_data_Y_3 = Y_3[30000:]
    validation_data_Y_4 = Y_4[30000:]
    validation_data_Y_5 = Y_5[30000:]

    train_tuple = zip(train_data_X, train_data_Y, train_data_Y_1, train_data_Y_2, train_data_Y_3, train_data_Y_4, train_data_Y_5)
    validation_tuple = zip(validation_data_X, validation_data_Y, validation_data_Y_1, validation_data_Y_2, validation_data_Y_3, validation_data_Y_4, validation_data_Y_5)


    for i in range(50000):
        batch = random.sample(train_tuple, 32)
        batch_data = [zz[0] for zz in batch]
        batch_labels = [zz[1] for zz in batch]
        batch_Y_1 = [zz[2] for zz in batch]
        batch_Y_2 = [zz[3] for zz in batch]
        batch_Y_3 = [zz[4] for zz in batch]
        batch_Y_4 = [zz[5] for zz in batch]
        batch_Y_5 = [zz[6] for zz in batch]


        if i%500==0:
            v_batch = random.sample(validation_tuple, 32)
            v_batch_data = [zz[0] for zz in v_batch]
            v_batch_labels = [zz[1] for zz in v_batch]
            v_batch_Y_1 = [zz[2] for zz in v_batch]
            v_batch_Y_2 = [zz[3] for zz in v_batch]
            v_batch_Y_3 = [zz[4] for zz in v_batch]
            v_batch_Y_4 = [zz[5] for zz in v_batch]
            v_batch_Y_5 = [zz[6] for zz in v_batch]

            v_acc = accuracy.eval(feed_dict={x: v_batch_data, y_: v_batch_labels, y_1: v_batch_Y_1, y_2: v_batch_Y_2, keep_prob: 1.0})
            print "validation acc = ", v_acc
            v_acc = accuracy_1.eval(feed_dict={x: v_batch_data, y_: v_batch_labels, y_1: v_batch_Y_1, y_2: v_batch_Y_2, keep_prob: 1.0})
            print "validation acc = ", v_acc
            v_acc = accuracy_2.eval(feed_dict={x: v_batch_data, y_: v_batch_labels, y_1: v_batch_Y_1, y_2: v_batch_Y_2, keep_prob: 1.0})
            print "validation acc = ", v_acc
            v_acc = accuracy_3.eval(feed_dict={x: v_batch_data, y_3: v_batch_Y_3, keep_prob: 1.0})
            print "validation acc = ", v_acc
            v_acc = accuracy_4.eval(feed_dict={x: v_batch_data, y_4: v_batch_Y_4, keep_prob: 1.0})
            print "validation acc = ", v_acc
            v_acc = accuracy_5.eval(feed_dict={x: v_batch_data, y_5: v_batch_Y_5, keep_prob: 1.0})
            print "validation acc = ", v_acc


        _, loss_val, acc, acc1, acc2, acc3, acc4, acc5 = sess.run([train_step, cross_entropy, accuracy, accuracy_1, accuracy_2, accuracy_3, accuracy_4, accuracy_5], feed_dict={x:batch_data, y_: batch_labels, keep_prob: 0.5, lr: 2e-4, y_1: batch_Y_1, y_2: batch_Y_2, y_3: batch_Y_3, y_4: batch_Y_4, y_5: batch_Y_5})

        if i%50 == 0:
            print "step", i, "loss", loss_val, "train_accuracy", acc, acc1, acc2, acc3, acc4, acc5

    va = 0
    for j in xrange(0, validation_data_X.shape[0], 8):
        mx = min(j+8, validation_data_X.shape[0])
        va = va + (accuracy.eval(feed_dict={x: validation_data_X[j:mx], y_: validation_data_Y[j:mx], y_1: validation_data_Y_1[j:mx], y_2: validation_data_Y_2[j:mx], keep_prob: 1.0}))*(mx-j)
    va /= validation_data_X.shape[0]
    print "accuracy", va

    va = 0
    for j in xrange(0, validation_data_X.shape[0], 8):
        mx = min(j+8, validation_data_X.shape[0])
        va = va + (accuracy_1.eval(feed_dict={x: validation_data_X[j:mx], y_: validation_data_Y[j:mx], y_1: validation_data_Y_1[j:mx], y_2: validation_data_Y_2[j:mx], keep_prob: 1.0}))*(mx-j)
    va /= validation_data_X.shape[0]
    print "accuracy", va

    va = 0
    for j in xrange(0, validation_data_X.shape[0], 8):
        mx = min(j+8, validation_data_X.shape[0])
        va = va + (accuracy_2.eval(feed_dict={x: validation_data_X[j:mx], y_: validation_data_Y[j:mx], y_1: validation_data_Y_1[j:mx], y_2: validation_data_Y_2[j:mx], keep_prob: 1.0}))*(mx-j)
    va /= validation_data_X.shape[0]
    print "accuracy", va

    va = 0
    for j in xrange(0, validation_data_X.shape[0], 8):
        mx = min(j+8, validation_data_X.shape[0])
        va = va + (accuracy_3.eval(feed_dict={x: validation_data_X[j:mx], y_: validation_data_Y[j:mx], y_1: validation_data_Y_1[j:mx], y_3: validation_data_Y_3[j:mx], keep_prob: 1.0}))*(mx-j)
    va /= validation_data_X.shape[0]
    print "accuracy", va


    va = 0
    for j in xrange(0, validation_data_X.shape[0], 8):
        mx = min(j+8, validation_data_X.shape[0])
        va = va + (accuracy_4.eval(feed_dict={x: validation_data_X[j:mx], y_: validation_data_Y[j:mx], y_1: validation_data_Y_1[j:mx], y_4: validation_data_Y_4[j:mx], keep_prob: 1.0}))*(mx-j)
    va /= validation_data_X.shape[0]
    print "accuracy", va


    va = 0
    for j in xrange(0, validation_data_X.shape[0], 8):
        mx = min(j+8, validation_data_X.shape[0])
        va = va + (accuracy_5.eval(feed_dict={x: validation_data_X[j:mx], y_: validation_data_Y[j:mx], y_1: validation_data_Y_1[j:mx], y_5: validation_data_Y_5[j:mx], keep_prob: 1.0}))*(mx-j)
    va /= validation_data_X.shape[0]
    print "accuracy", va

    saver.save(sess, './model1.ckpt')

def load_model():
    saver.restore(sess, './model1.ckpt')

def get_predictions(X):
    y, y1, y2, y3, y4, y5 = sess.run([arg_y, arg_y_1, arg_y_2, arg_y_3, arg_y_4, arg_y_5], feed_dict={x:[X], keep_prob: 1.0})
    return [y[0], y1[0], y2[0], y3[0], y4[0], y5[0]]
