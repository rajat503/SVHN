import numpy as np
import tensorflow as tf
import random

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 5])
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
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# W_fc2 = weight_variable([1024, 1024])
# b_fc2 = bias_variable([1024])
# h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([1024, 1024])
b_fc3 = bias_variable([1024])
h_fc3 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

W_fc4 = weight_variable([1024, 5])
b_fc4 = bias_variable([5])
y_conv=tf.nn.softmax(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

X = np.load('dump/cropped_images')
Y = np.load('dump/one_hot_sequence_len')

train_data_X = X[:30000]
train_data_Y = Y[:30000]

validation_data_X = X[30000:]
validation_data_Y = Y[30000:]

train_tuple = zip(train_data_X, train_data_Y)
validation_tuple = zip(validation_data_X, validation_data_Y)


for i in range(10000):
    batch = random.sample(train_tuple, 64)
    batch_data = [zz[0] for zz in batch]
    batch_labels = [zz[1] for zz in batch]

    if i%500==0:
        v_batch = random.sample(validation_tuple, 64)
        v_batch_data = [zz[0] for zz in v_batch]
        v_batch_labels = [zz[1] for zz in v_batch]
        v_acc = accuracy.eval(feed_dict={x: v_batch_data, y_: v_batch_labels, keep_prob: 1.0})
        print "validation acc = ", v_acc

    _, loss_val, acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={x:batch_data, y_: batch_labels, keep_prob: 0.5, lr: 2e-4})

    if i%50 == 0:
        print "step", i, "loss", loss_val, "train_accuracy", acc

va = 0
for j in xrange(0, validation_data_X.shape[0], 8):
    mx = min(j+8, validation_data_X.shape[0])
    va = va + (accuracy.eval(feed_dict={x: validation_data_X[j:mx], y_: validation_data_Y[j:mx], keep_prob: 1.0}))*(mx-j)
va /= validation_data_X.shape[0]
print "accuracy", va
