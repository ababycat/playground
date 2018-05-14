#!usr/bin/env python3

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

class resnet:
    def __init__(self):
        return

class gauss_model:
    def __init__(self):
        self.train_loss_log = []
        self.valid_loss_log = []
        self.test_loss_log = []
        self.train_accuracy_log = []  
        self.valid_accuracy_log = []
        self.test_accuracy_log = []

        self.momentum = 0.997
        self.global_step = None
        self.learning_rate = None

        self.update_op = []
        self.all_loss_real_list, self.all_accuracy_real_list = [], []

    def gauss(self, x, y, mu1, mu2, sigma1, sigma2, ro):
        return 1/(2*np.pi*sigma1*sigma2*tf.sqrt(1-ro**2)) * tf.exp( -1/(2*(1-ro**2)) * ( ((x-mu1)/sigma1)**2 - 2*ro*(x-mu1)*(y-mu2)/(sigma1*sigma2+1e-3) + ((y-mu2)/(sigma2+1e-3))**2  ))

    def _get_conv_bn_param(self, filter_size=3, output_channels=64, input_channels=3, W_stddev = 0.001, isAddW2Loss=True, dtype=tf.float32):
        tf.get_variable('W', shape=[filter_size,filter_size,input_channels,output_channels], 
                        dtype=dtype, initializer=tf.random_normal_initializer(mean=0.0, stddev=W_stddev))
        tf.get_variable('b', shape=[output_channels],
                            dtype=dtype, initializer=tf.zeros_initializer(dtype=dtype))
        tf.get_variable('scale', shape=[1,1,1,output_channels], 
                            dtype=dtype, initializer=tf.ones_initializer(dtype=dtype))
        tf.get_variable('offset', shape=[1,1,1,output_channels], 
                            dtype=dtype, initializer=tf.zeros_initializer(dtype=dtype))
        tf.get_variable('mean', shape=[1,1,1,output_channels], 
                            dtype=dtype, initializer=tf.zeros_initializer(dtype=dtype), trainable=False)
        tf.get_variable('var', shape=[1,1,1,output_channels], 
                            dtype=dtype, initializer=tf.ones_initializer(dtype=dtype), trainable=False)
        if isAddW2Loss:
            tf.get_variable_scope().reuse_variables()
            tf.add_to_collection('loss_W', tf.reduce_sum(tf.get_variable('W')**2))

    def _get_conv_param(self, filter_size=3, output_channels=64, input_channels=3, W_stddev = 0.001, isAddW2Loss=True, dtype=tf.float32):
        tf.get_variable('W', shape=[filter_size,filter_size,input_channels, output_channels], 
                        dtype=dtype, initializer=tf.random_normal_initializer(mean=0.0, stddev=W_stddev))
        tf.get_variable('b', shape=[output_channels],
                            dtype=dtype, initializer=tf.zeros_initializer(dtype=dtype))
        if isAddW2Loss:
            tf.get_variable_scope().reuse_variables()
            tf.add_to_collection('loss_W', tf.reduce_sum(tf.get_variable('W')**2))
    
    def get_gauss_variable(self):
        with tf.variable_scope('gauss'):
            rows = 28
            cols = 28
            x = np.ones((rows,cols), dtype=np.float32)
            y = np.ones((rows,cols), dtype=np.float32)
            X = np.diag(np.arange(rows)).dot(x)
            Y = y.dot(np.diag(np.arange(cols)))
            X = ((X - np.mean(np.arange(rows)))/np.mean(np.arange(rows))).astype(np.float32)
            Y = ((Y - np.mean(np.arange(cols)))/np.mean(np.arange(cols))).astype(np.float32)
            tf.get_variable('gauss_X', initializer=tf.constant(X.reshape(28,28,1)), dtype=tf.float32, trainable=False)
            tf.get_variable('gauss_Y', initializer=tf.constant(Y.reshape(28,28,1)), dtype=tf.float32, trainable=False)
        with tf.variable_scope('calculate_mu1'):
            with tf.variable_scope('conv1'):
                self._get_conv_bn_param(filter_size=3, output_channels=32, input_channels=1, W_stddev = 0.001, isAddW2Loss=True)
            with tf.variable_scope('conv2'):
                self._get_conv_bn_param(filter_size=3, output_channels=64, input_channels=32, W_stddev = 0.001)
            with tf.variable_scope('conv3'):
                self._get_conv_bn_param(filter_size=3, output_channels=128, input_channels=64, W_stddev = 0.01)
            with tf.variable_scope('conv4'):
                self._get_conv_bn_param(filter_size=1, output_channels=256, input_channels=128, W_stddev = 0.1)
            with tf.variable_scope('conv5'):
                self._get_conv_param(filter_size=1, output_channels=1, input_channels=256, W_stddev = 0.1)
        with tf.variable_scope('calculate_mu2'):
            with tf.variable_scope('conv1'):
                self._get_conv_bn_param(filter_size=3, output_channels=32, input_channels=1, W_stddev = 0.001, isAddW2Loss=True)
            with tf.variable_scope('conv2'):
                self._get_conv_bn_param(filter_size=3, output_channels=64, input_channels=32, W_stddev = 0.001)
            with tf.variable_scope('conv3'):
                self._get_conv_bn_param(filter_size=3, output_channels=128, input_channels=64, W_stddev = 0.01)
            with tf.variable_scope('conv4'):
                self._get_conv_bn_param(filter_size=1, output_channels=256, input_channels=128, W_stddev = 0.1)
            with tf.variable_scope('conv5'):
                self._get_conv_param(filter_size=1, output_channels=1, input_channels=256, W_stddev = 0.1)
        with tf.variable_scope('calculate_sigma1'):
            with tf.variable_scope('conv1'):
                self._get_conv_bn_param(filter_size=3, output_channels=32, input_channels=1, W_stddev = 0.001, isAddW2Loss=True)
            with tf.variable_scope('conv2'):
                self._get_conv_bn_param(filter_size=3, output_channels=64, input_channels=32, W_stddev = 0.001)
            with tf.variable_scope('conv3'):
                self._get_conv_bn_param(filter_size=3, output_channels=128, input_channels=64, W_stddev = 0.01)
            with tf.variable_scope('conv4'):
                self._get_conv_bn_param(filter_size=1, output_channels=256, input_channels=128, W_stddev = 0.1)
            with tf.variable_scope('conv5'):
                self._get_conv_param(filter_size=1, output_channels=1, input_channels=256, W_stddev = 0.1)
        with tf.variable_scope('calculate_sigma2'):
            with tf.variable_scope('conv1'):
                self._get_conv_bn_param(filter_size=3, output_channels=32, input_channels=1, W_stddev = 0.001, isAddW2Loss=True)
            with tf.variable_scope('conv2'):
                self._get_conv_bn_param(filter_size=3, output_channels=64, input_channels=32, W_stddev = 0.001)
            with tf.variable_scope('conv3'):
                self._get_conv_bn_param(filter_size=3, output_channels=128, input_channels=64, W_stddev = 0.01)
            with tf.variable_scope('conv4'):
                self._get_conv_bn_param(filter_size=1, output_channels=256, input_channels=128, W_stddev = 0.1)
            with tf.variable_scope('conv5'):
                self._get_conv_param(filter_size=1, output_channels=1, input_channels=256, W_stddev = 0.1)
        with tf.variable_scope('calculate_ro'):
            with tf.variable_scope('conv1'):
                self._get_conv_bn_param(filter_size=3, output_channels=32, input_channels=1, W_stddev = 0.001, isAddW2Loss=True)
            with tf.variable_scope('conv2'):
                self._get_conv_bn_param(filter_size=3, output_channels=64, input_channels=32, W_stddev = 0.001)
            with tf.variable_scope('conv3'):
                self._get_conv_bn_param(filter_size=3, output_channels=128, input_channels=64, W_stddev = 0.01)
            with tf.variable_scope('conv4'):
                self._get_conv_bn_param(filter_size=1, output_channels=256, input_channels=128, W_stddev = 0.1)
            with tf.variable_scope('conv5'):
                self._get_conv_param(filter_size=1, output_channels=1, input_channels=256, W_stddev = 0.1)
        # H, W, C, F
        with tf.variable_scope('forward'):
            with tf.variable_scope('conv1'):
                self._get_conv_bn_param(filter_size=3, output_channels=32, input_channels=1, W_stddev = 0.001, isAddW2Loss=True)
            with tf.variable_scope('conv2'):
                self._get_conv_bn_param(filter_size=3, output_channels=64, input_channels=32, W_stddev = 0.001)
            with tf.variable_scope('conv2_neck'):
                self._get_conv_bn_param(filter_size=3, output_channels=1, input_channels=64, W_stddev = 0.001)
            with tf.variable_scope('conv3'):
                self._get_conv_bn_param(filter_size=3, output_channels=64, input_channels=1, W_stddev = 0.01)
            with tf.variable_scope('conv4'):
                self._get_conv_bn_param(filter_size=1, output_channels=256, input_channels=64, W_stddev = 0.1)
            with tf.variable_scope('conv5'):
                self._get_conv_param(filter_size=1, output_channels=10, input_channels=256, W_stddev = 0.1)

    def forward(self, X, isTraining):
        with tf.variable_scope('calculate_mu1'):
            with tf.variable_scope('conv1', reuse=True):
                y = self.conv_bn_relu_pool(X, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv2', reuse=True):
                y = self.conv_bn_relu_pool(y, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv3', reuse=True):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
                y = tf.nn.avg_pool(y, ksize=[1,4,4,1], strides=[1,1,1,1], padding='VALID')
            with tf.variable_scope('conv4', reuse=True):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('conv5', reuse=True):
                y = tf.nn.conv2d(y, filter=tf.get_variable('W'), strides=[1,1,1,1], padding='VALID')
                y = tf.nn.tanh(y)
                mu1 = tf.reshape(y, shape=[-1,1,1,1])
        with tf.variable_scope('calculate_mu2'):
            with tf.variable_scope('conv1', reuse=True):
                y = self.conv_bn_relu_pool(X, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv2', reuse=True):
                y = self.conv_bn_relu_pool(y, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv3', reuse=True):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
                y = tf.nn.avg_pool(y, ksize=[1,4,4,1], strides=[1,1,1,1], padding='VALID')
            with tf.variable_scope('conv4', reuse=True):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('conv5', reuse=True):
                y = tf.nn.conv2d(y, filter=tf.get_variable('W'), strides=[1,1,1,1], padding='VALID')
                y = tf.nn.tanh(y)
                mu2 = tf.reshape(y, shape=[-1,1,1,1])
        with tf.variable_scope('calculate_sigma1'):
            with tf.variable_scope('conv1', reuse=True):
                y = self.conv_bn_relu_pool(X, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv2', reuse=True):
                y = self.conv_bn_relu_pool(y, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv3', reuse=True):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
                y = tf.nn.avg_pool(y, ksize=[1,4,4,1], strides=[1,1,1,1], padding='VALID')
            with tf.variable_scope('conv4', reuse=True):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('conv5', reuse=True):
                y = tf.nn.conv2d(y, filter=tf.get_variable('W'), strides=[1,1,1,1], padding='VALID')
                y = tf.nn.sigmoid(y)+1e-8
                sigma1 = tf.reshape(y, shape=[-1,1,1,1])
        with tf.variable_scope('calculate_sigma2'):
            with tf.variable_scope('conv1', reuse=True):
                y = self.conv_bn_relu_pool(X, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv2', reuse=True):
                y = self.conv_bn_relu_pool(y, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv3', reuse=True):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
                y = tf.nn.avg_pool(y, ksize=[1,4,4,1], strides=[1,1,1,1], padding='VALID')
            with tf.variable_scope('conv4', reuse=True):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('conv5', reuse=True):
                y = tf.nn.conv2d(y, filter=tf.get_variable('W'), strides=[1,1,1,1], padding='VALID')
                y = tf.nn.sigmoid(y)+1e-8
                sigma2 = tf.reshape(y, shape=[-1,1,1,1])
        with tf.variable_scope('calculate_ro'):
            with tf.variable_scope('conv1', reuse=True):
                y = self.conv_bn_relu_pool(X, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv2', reuse=True):
                y = self.conv_bn_relu_pool(y, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv3', reuse=True):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
                y = tf.nn.avg_pool(y, ksize=[1,4,4,1], strides=[1,1,1,1], padding='VALID')
            with tf.variable_scope('conv4', reuse=True):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('conv5', reuse=True):
                y = tf.nn.conv2d(y, filter=tf.get_variable('W'), strides=[1,1,1,1], padding='VALID')
                y = tf.nn.tanh(y)
                ro = tf.reshape(y, shape=[-1,1,1,1])
        with tf.variable_scope('forward', reuse=True):
            with tf.variable_scope('conv1', reuse=True):
                y = self.conv_bn_relu(X, isTraining, conv_strides=1, conv_padding='SAME')
            with tf.variable_scope('conv2', reuse=True):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='SAME')
            with tf.variable_scope('conv2_neck'):
                y = self.conv_bn(y, isTraining, conv_strides=1, conv_padding='SAME')
        with tf.variable_scope('gauss', reuse=True):
            p = self.gauss(tf.get_variable('gauss_X'), tf.get_variable('gauss_Y'), mu1, mu2, sigma1, sigma2, ro)
            y =  y * p
        with tf.variable_scope('forward', reuse=True):
            with tf.variable_scope('conv3', reuse=True):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
                y = tf.nn.avg_pool(y, ksize=[1,26,26,1], strides=[1,1,1,1], padding='VALID')
            with tf.variable_scope('conv4', reuse=True):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('conv5', reuse=True):
                y = tf.nn.conv2d(y, filter=tf.get_variable('W'), strides=[1,1,1,1], padding='VALID')
                y = tf.reshape(y, shape=[-1,10])
        return y, p

    def conv_bn_relu_pool(self, X, isTraining, conv_strides=1, conv_padding='SAME', pool_size=2, pool_strides=2, pool_padding='SAME'):
        Z1 = tf.nn.conv2d(X, tf.get_variable('W'), strides=[1,conv_strides, conv_strides, 1], padding=conv_padding, name='conv')        
        if isTraining:
            M1, V1 = tf.nn.moments(Z1, axes=[0,1,2], shift=1e-8, keep_dims=True, name='moments')
            running_var_op = tf.assign(tf.get_variable('var'), (self.momentum*V1 + (1-self.momentum)*V1), name='running_var')
            self.update_op.append(running_var_op)
            running_mean_op = tf.assign(tf.get_variable('mean'), (self.momentum*M1 + (1-self.momentum)*M1), name='running_mean')
            self.update_op.append(running_mean_op)
        else:
            M1, V1 = tf.get_variable('mean'), tf.get_variable('var')
        scale1, offset1 = tf.get_variable('scale'), tf.get_variable('offset')
        B1 = tf.nn.batch_normalization(Z1, M1, V1, offset1, scale1, variance_epsilon=1e-8, name='bn')
        A1 = tf.nn.relu(B1, name='relu')
        P1 = tf.nn.max_pool(A1, ksize=[1,pool_size,pool_size,1], strides=[1,pool_strides,pool_strides,1], padding=pool_padding, name='pool')
        return P1

    def conv_bn_relu(self, X, isTraining, conv_strides=1, conv_padding='SAME'):
        Z1 = tf.nn.conv2d(X, tf.get_variable('W'), strides=[1,conv_strides, conv_strides, 1], padding=conv_padding, name='conv')        
        if isTraining:
            M1, V1 = tf.nn.moments(Z1, axes=[0,1,2], shift=1e-8, keep_dims=True, name='moments')
            running_var_op = tf.assign(tf.get_variable('var'), (self.momentum*V1 + (1-self.momentum)*V1), name='running_var')
            self.update_op.append(running_var_op)
            running_mean_op = tf.assign(tf.get_variable('mean'), (self.momentum*M1 + (1-self.momentum)*M1), name='running_mean')
            self.update_op.append(running_mean_op)
        else:
            M1, V1 = tf.get_variable('mean'), tf.get_variable('var')
        scale1, offset1 = tf.get_variable('scale'), tf.get_variable('offset')
        B1 = tf.nn.batch_normalization(Z1, M1, V1, offset1, scale1, variance_epsilon=1e-8, name='bn')
        A1 = tf.nn.relu(B1, name='relu')
        return A1

    def conv_bn(self, X, isTraining, conv_strides=1, conv_padding='SAME'):
        Z1 = tf.nn.conv2d(X, tf.get_variable('W'), strides=[1,conv_strides, conv_strides, 1], padding=conv_padding, name='conv')        
        if isTraining:
            M1, V1 = tf.nn.moments(Z1, axes=[0,1,2], shift=1e-8, keep_dims=True, name='moments')
            running_var_op = tf.assign(tf.get_variable('var'), (self.momentum*V1 + (1-self.momentum)*V1), name='running_var')
            self.update_op.append(running_var_op)
            running_mean_op = tf.assign(tf.get_variable('mean'), (self.momentum*M1 + (1-self.momentum)*M1), name='running_mean')
            self.update_op.append(running_mean_op)
        else:
            M1, V1 = tf.get_variable('mean'), tf.get_variable('var')
        scale1, offset1 = tf.get_variable('scale'), tf.get_variable('offset')
        B1 = tf.nn.batch_normalization(Z1, M1, V1, offset1, scale1, variance_epsilon=1e-8, name='bn')
        return B1
    
    def loss(self, y_out, y, reg):
        mean_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), y_out)) + reg*tf.add_n(tf.get_collection('loss_W'))
        return mean_loss

    def accuracy(self, y_out, y):
        correct_prediction = tf.equal(tf.cast(tf.argmax(y_out,1), tf.int32), y)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc

    def predict_cnn(self, sess, data, X_y_holder):
        batch_size = 128
        isTraining = False
        Z = self.forward(X_y_holder[0], isTraining)
        # loss = self.loss(Z, X_y_holder[1], reg)
        accuracy = self.accuracy(Z, X_y_holder[1])
        
        X_eval, y_eval = data
        
        m = np.shape(X_eval)[0]
        loss_real_list, accuracy_real_list = [], []
        tm = m//batch_size + 1
        for i in range(tm):
            X_batch = X_eval[i*batch_size:i*batch_size+batch_size,:]
            y_batch = y_eval[i*batch_size:i*batch_size+batch_size]
            feed_dict = {X_y_holder[0]:X_batch, X_y_holder[1]:y_batch}
            accuracy_real_batch = sess.run([accuracy], feed_dict=feed_dict)
            # loss_real_list.append(loss_real_batch)
            accuracy_real_list.append(accuracy_real_batch)
        _, accuracy_real = np.sum(loss_real_list)/tm, np.sum(accuracy_real_list)/tm
        return accuracy_real

    def evaluate(self, sess, data, X_y_holder, loss, accuracy, batch_size):
        X_eval, y_eval = data
        m = np.shape(X_eval)[0]
        loss_real_list, accuracy_real_list = [], []
        tm = m//batch_size + 1
        for i in range(tm):
            X_batch = X_eval[i*batch_size:i*batch_size+batch_size,:]
            y_batch = y_eval[i*batch_size:i*batch_size+batch_size]
            feed_dict = {X_y_holder[0]:X_batch, X_y_holder[1]:y_batch}
            loss_real_batch, accuracy_real_batch = sess.run([loss, accuracy], feed_dict=feed_dict)
            loss_real_list.append(loss_real_batch)
            accuracy_real_list.append(accuracy_real_batch)
        loss_real, accuracy_real = np.sum(loss_real_list)/tm, np.sum(accuracy_real_list)/tm
        return loss_real, accuracy_real

    def train(self, sess, data, X_y_holder, loss, accuracy, optimizer, batch_size, p):
        X_eval, y_eval = data
        m = np.shape(X_eval)[0]
        loss_real_list, accuracy_real_list = [], []
        tm = m//batch_size + 1
        for i in range(tm):
        # for i in range(20):
            X_batch = X_eval[i*batch_size:i*batch_size+batch_size,:]
            y_batch = y_eval[i*batch_size:i*batch_size+batch_size]
            feed_dict = {X_y_holder[0]:X_batch, X_y_holder[1]:y_batch}
#             print(sess.run(p, feed_dict=feed_dict))
            loss_real_batch, accuracy_real_batch, _ = sess.run([loss, accuracy, optimizer], feed_dict=feed_dict)
#             print(loss_real_batch)
            loss_real_list.append(loss_real_batch)
            accuracy_real_list.append(accuracy_real_batch)
        loss_real, accuracy_real = np.sum(loss_real_list)/tm, np.sum(accuracy_real_list)/tm
        return loss_real, accuracy_real

    def do_it(self, sess, data, epchos, batch_size, reg, starter_learning_rate, decay_steps, decay_rate):
        X = tf.placeholder(dtype=tf.float32, shape=[None,28,28,1],name='X')
        y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')
        isTraining = True
        Z_train, p = self.forward(X, isTraining)
        loss_train = self.loss(Z_train, y, reg)
        accuracy_train = self.accuracy(Z_train, y)

        isTraining = False
        Z_val_test, p = self.forward(X, isTraining)
        loss_val_test = self.loss(Z_val_test, y, reg)
        accuracy_val_test = self.accuracy(Z_val_test, y)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                decay_steps, decay_rate, staircase=True, name='learning_rate')
        
        AdamOptimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step = AdamOptimizer.minimize(loss_train, global_step=self.global_step)
        optimizer = [self.update_op, train_step]

        # # Scale the learning rate linearly with the batch size. When the batch size
        # # is 128, the learning rate should be 0.1.
        # initial_learning_rate = 0.1 * batch_size / 128
        # batches_per_epoch = 50000 / batch_size
        # self.global_step = tf.train.get_or_create_global_step()

        # # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        # boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
        # values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
        # self.learning_rate = tf.train.piecewise_constant(
        #     tf.cast(global_step, tf.int32), boundaries, values)

        sess.run(tf.global_variables_initializer())

        X_y_holder = (X, y)
        self.run(sess, data, X_y_holder, loss_train, accuracy_train, loss_val_test, accuracy_val_test, optimizer, batch_size=batch_size, epchos=epchos, p=p)

    def run(self, sess, data, X_y_holder, loss_train, accuracy_train, loss_val_test, accuracy_val_test, optimizer, batch_size, epchos, p):
        """X_y_holder: (X, y)
        data = (X_train, y_train, X_valid, y_valid, X_test, y_test)
        """
        X_train, y_train, X_valid, y_valid, X_test, y_test = data
        data_train = (X_train, y_train)
        data_valid = (X_valid, y_valid)
        data_test = (X_test, y_test)
        
        loss_real_train, accuracy_real_train = self.evaluate(sess, data_train, X_y_holder, loss_val_test, accuracy_val_test, batch_size)
        loss_real_valid, accuracy_real_valid = self.evaluate(sess, data_valid, X_y_holder, loss_val_test, accuracy_val_test, batch_size)
        loss_real_test, accuracy_real_test = self.evaluate(sess, data_test, X_y_holder, loss_val_test, accuracy_val_test, batch_size)

        self.train_loss_log.append(loss_real_train)
        self.train_accuracy_log.append(accuracy_real_train)
        self.test_loss_log.append(loss_real_test)
        self.test_accuracy_log.append(accuracy_real_test)
        self.valid_loss_log.append(loss_real_valid)
        self.valid_accuracy_log.append(accuracy_real_valid)
        print('epoch %d Loss: train %0.5f, valid %0.5f, test %0.5f, acc: train %0.5f, valid %0.5f, test %0.5f'%(
                0, loss_real_train, loss_real_valid, loss_real_test, accuracy_real_train, accuracy_real_valid, accuracy_real_test))

        for epcho in range(epchos):
            X_train, y_train = data_train
            idx = np.random.permutation(X_train.shape[0])
            X_train_tmp = X_train[idx, :]
            y_train_tmp = y_train[idx]
            data_train = (X_train_tmp, y_train_tmp)
            loss_real_train, accuracy_real_train = self.train(sess, data_train, X_y_holder, loss_train, accuracy_train, optimizer, batch_size, p)
            loss_real_valid, accuracy_real_valid = self.evaluate(sess, data_valid, X_y_holder, loss_val_test, accuracy_val_test, batch_size)
            loss_real_test, accuracy_real_test = self.evaluate(sess, data_test, X_y_holder, loss_val_test, accuracy_val_test, batch_size)
            self.train_loss_log.append(loss_real_train)
            self.train_accuracy_log.append(accuracy_real_train)
            self.test_loss_log.append(loss_real_test)
            self.test_accuracy_log.append(accuracy_real_test)
            self.valid_loss_log.append(loss_real_valid)
            self.valid_accuracy_log.append(accuracy_real_valid)
            
            with open('min_loss1.txt', 'r+') as fp:
                line = fp.readline()
                tmp_acc = float(line)
                if accuracy_real_test > tmp_acc:
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, "model.ckpt")
                    print("Model saved in file: %s" % save_path)
                    fp.seek(0,0)
                    fp.truncate()
                    fp.write(str(accuracy_real_test))

            print('epoch %d Loss: train %0.5f, valid %0.5f, test %0.5f, acc: train %0.5f, valid %0.5f, test %0.5f'%(
                    epcho+1, loss_real_train, loss_real_valid, loss_real_test, accuracy_real_train, accuracy_real_valid, accuracy_real_test))
            print('global step %d, learning rate %f'%(sess.run(self.global_step), sess.run(self.learning_rate)))


class model:
    def __init__(self):
        self.train_loss_log = []
        self.valid_loss_log = []
        self.test_loss_log = []
        self.train_accuracy_log = []  
        self.valid_accuracy_log = []
        self.test_accuracy_log = []

        self.momentum = 0.997
        self.global_step = None
        self.learning_rate = None

        self.update_op = []
        
    def _get_conv_bn_param(self, filter_size=3, output_channels=64, input_channels=3, W_stddev = 0.001, isAddW2Loss=True, dtype=tf.float32):
        tf.get_variable('W', shape=[filter_size,filter_size,input_channels,output_channels], 
                        dtype=dtype, initializer=tf.random_normal_initializer(mean=0.0, stddev=W_stddev))
        tf.get_variable('b', shape=[output_channels],
                            dtype=dtype, initializer=tf.zeros_initializer(dtype=dtype))
        tf.get_variable('scale', shape=[1,1,1,output_channels], 
                            dtype=dtype, initializer=tf.ones_initializer(dtype=dtype))
        tf.get_variable('offset', shape=[1,1,1,output_channels], 
                            dtype=dtype, initializer=tf.zeros_initializer(dtype=dtype))
        tf.get_variable('mean', shape=[1,1,1,output_channels], 
                            dtype=dtype, initializer=tf.zeros_initializer(dtype=dtype), trainable=False)
        tf.get_variable('var', shape=[1,1,1,output_channels], 
                            dtype=dtype, initializer=tf.ones_initializer(dtype=dtype), trainable=False)
        if isAddW2Loss:
            tf.get_variable_scope().reuse_variables()
            tf.add_to_collection('loss_W', tf.reduce_sum(tf.get_variable('W')**2))

    def _get_conv_param(self, filter_size=3, output_channels=64, input_channels=3, W_stddev = 0.001, isAddW2Loss=True, dtype=tf.float32):
        tf.get_variable('W', shape=[filter_size,filter_size,input_channels, output_channels], 
                        dtype=dtype, initializer=tf.random_normal_initializer(mean=0.0, stddev=W_stddev))
        tf.get_variable('b', shape=[output_channels],
                            dtype=dtype, initializer=tf.zeros_initializer(dtype=dtype))
        if isAddW2Loss:
            tf.get_variable_scope().reuse_variables()
            tf.add_to_collection('loss_W', tf.reduce_sum(tf.get_variable('W')**2))
    
    def get_gauss_variable(sefl):
        with tf.variable_scope('gauss'):
            rows = 28
            cols = 28
            x = np.ones((rows,cols))
            y = np.ones((rows,cols))
            X = np.diag(np.arange(rows)).dot(x)
            Y = y.dot(np.diag(np.arange(cols)))
            X = (X - np.mean(np.arange(rows)))/np.mean(np.arange(rows))
            Y = (Y - np.mean(np.arange(cols)))/np.mean(np.arange(cols))
            tf.get_variable('gauss_X', initializer=tf.constant(X))
            tf.get_variable('gauss_Y', initializer=tf.constant(Y))
        # H, W, C, F
        with tf.variable_scope('conv1'):
            self._get_conv_bn_param(filter_size=3, output_channels=32, input_channels=1, W_stddev = 0.001, isAddW2Loss=True)
        with tf.variable_scope('conv2'):
            self._get_conv_bn_param(filter_size=3, output_channels=64, input_channels=32, W_stddev = 0.001)
        with tf.variable_scope('conv3'):
            self._get_conv_bn_param(filter_size=3, output_channels=128, input_channels=64, W_stddev = 0.01)
        with tf.variable_scope('conv4'):
            self._get_conv_bn_param(filter_size=1, output_channels=256, input_channels=128, W_stddev = 0.1)
        with tf.variable_scope('conv5'):
            self._get_conv_param(filter_size=1, output_channels=10, input_channels=256, W_stddev = 0.1)

    def get_varibale(self, dtype=tf.float32):
        # H, W, C, F
        with tf.variable_scope('conv1'):
            self._get_conv_bn_param(filter_size=3, output_channels=32, input_channels=1, W_stddev = 0.001, isAddW2Loss=True)
        with tf.variable_scope('conv2'):
            self._get_conv_bn_param(filter_size=3, output_channels=64, input_channels=32, W_stddev = 0.001)
        with tf.variable_scope('conv3'):
            self._get_conv_bn_param(filter_size=3, output_channels=128, input_channels=64, W_stddev = 0.01)
        with tf.variable_scope('conv4'):
            tf.get_variable('W', shape=[128,10], dtype=tf.float32)
            tf.get_variable('b', shape=[10], dtype=tf.float32)
        # with tf.variable_scope('conv4'):
        #     self._get_conv_bn_param(filter_size=1, output_channels=256, input_channels=128, W_stddev = 0.1)
        # with tf.variable_scope('conv5'):
        #     self._get_conv_param(filter_size=1, output_channels=10, input_channels=256, W_stddev = 0.1)
        
    def conv_bn_relu_pool(self, X, isTraining, conv_strides=1, conv_padding='SAME', pool_size=2, pool_strides=2, pool_padding='SAME'):
        Z1 = tf.nn.conv2d(X, tf.get_variable('W'), strides=[1,conv_strides, conv_strides, 1], padding=conv_padding, name='conv')        
        if isTraining:
            M1, V1 = tf.nn.moments(Z1, axes=[0,1,2], shift=1e-8, keep_dims=True, name='moments')
            running_var_op = tf.assign(tf.get_variable('var'), (self.momentum*V1 + (1-self.momentum)*V1), name='running_var')
            self.update_op.append(running_var_op)
            running_mean_op = tf.assign(tf.get_variable('mean'), (self.momentum*M1 + (1-self.momentum)*M1), name='running_mean')
            self.update_op.append(running_mean_op)
        else:
            M1, V1 = tf.get_variable('mean'), tf.get_variable('var')
        scale1, offset1 = tf.get_variable('scale'), tf.get_variable('offset')
        B1 = tf.nn.batch_normalization(Z1, M1, V1, offset1, scale1, variance_epsilon=1e-8, name='bn')
        A1 = tf.nn.relu(B1, name='relu')
        P1 = tf.nn.max_pool(A1, ksize=[1,pool_size,pool_size,1], strides=[1,pool_strides,pool_strides,1], padding=pool_padding, name='pool')
        return P1

    def conv_bn_relu(self, X, isTraining, conv_strides=1, conv_padding='SAME'):
        Z1 = tf.nn.conv2d(X, tf.get_variable('W'), strides=[1,conv_strides, conv_strides, 1], padding=conv_padding, name='conv')        
        if isTraining:
            M1, V1 = tf.nn.moments(Z1, axes=[0,1,2], shift=1e-8, keep_dims=True, name='moments')
            running_var_op = tf.assign(tf.get_variable('var'), (self.momentum*V1 + (1-self.momentum)*V1), name='running_var')
            self.update_op.append(running_var_op)
            running_mean_op = tf.assign(tf.get_variable('mean'), (self.momentum*M1 + (1-self.momentum)*M1), name='running_mean')
            self.update_op.append(running_mean_op)
        else:
            M1, V1 = tf.get_variable('mean'), tf.get_variable('var')
        scale1, offset1 = tf.get_variable('scale'), tf.get_variable('offset')
        B1 = tf.nn.batch_normalization(Z1, M1, V1, offset1, scale1, variance_epsilon=1e-8, name='bn')
        A1 = tf.nn.relu(B1, name='relu')
        return A1

    def conv_bn(self, X, isTraining, conv_strides=1, conv_padding='SAME'):
        Z1 = tf.nn.conv2d(X, tf.get_variable('W'), strides=[1,conv_strides, conv_strides, 1], padding=conv_padding, name='conv')        
        if isTraining:
            M1, V1 = tf.nn.moments(Z1, axes=[0,1,2], shift=1e-8, keep_dims=True, name='moments')
            running_var_op = tf.assign(tf.get_variable('var'), (self.momentum*V1 + (1-self.momentum)*V1), name='running_var')
            self.update_op.append(running_var_op)
            running_mean_op = tf.assign(tf.get_variable('mean'), (self.momentum*M1 + (1-self.momentum)*M1), name='running_mean')
            self.update_op.append(running_mean_op)
        else:
            M1, V1 = tf.get_variable('mean'), tf.get_variable('var')
        scale1, offset1 = tf.get_variable('scale'), tf.get_variable('offset')
        B1 = tf.nn.batch_normalization(Z1, M1, V1, offset1, scale1, variance_epsilon=1e-8, name='bn')
        return B1

    def forward(self, X, isTraining):
        with tf.variable_scope('conv1', reuse=True):
            y = self.conv_bn_relu_pool(X, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            print(y)
        with tf.variable_scope('conv2', reuse=True):
            y = self.conv_bn_relu_pool(y, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            print(y)
        with tf.variable_scope('conv3', reuse=True):
            y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
            print(y)
            y = tf.nn.avg_pool(y, ksize=[1,4,4,1], strides=[1,1,1,1], padding='VALID')
            print(y)
            y = tf.reshape(y, shape=[-1,128])
        with tf.variable_scope('conv4', reuse=True):
            y = tf.matmul(y, tf.get_variable('W')) + tf.get_variable('b')
        # with tf.variable_scope('conv4', reuse=True):
        #     y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
        #     # print(y)
        # with tf.variable_scope('conv5', reuse=True):
        #     y = tf.nn.conv2d(y, filter=tf.get_variable('W'), strides=[1,1,1,1], padding='VALID')
        #     y = tf.reshape(y, shape=[-1,10])
        return y

    def get_resnet_indentity_block_skip_3_param(self, name, filter_size, input_channels, filters_num=(64, 64, 64), dtype=tf.float32, isAddW2Loss=True):
        F1, F2, F3 = filters_num
        with tf.variable_scope(name):
            with tf.variable_scope('a'):
                self._get_conv_bn_param(1, F1, input_channels, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('b'):
                self._get_conv_bn_param(filter_size, F2, F1, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('c'):
                self._get_conv_bn_param(1, F3, F2, isAddW2Loss=isAddW2Loss)

    def get_resnet_conv_block_skip_3(self, name, filter_size, input_channels, filters_num=(64, 64, 64), dtype=tf.float32, isAddW2Loss=True):
        F1, F2, F3 = filters_num
        with tf.variable_scope(name):
            with tf.variable_scope('a'):
                self._get_conv_bn_param(1, F1, input_channels, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('b'):
                self._get_conv_bn_param(filter_size, F2, F1, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('c'):
                self._get_conv_bn_param(1, F3, F2, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('d'):
                self._get_conv_bn_param(1, F3, input_channels, isAddW2Loss=isAddW2Loss)

    # def get_resnet_param(self):
    #     with tf.variable_scope('stage1'):
    #         self._get_conv_bn_param(filter_size=7, output_channels=64, input_channels=3, isAddW2Loss=True)
    #     with tf.variable_scope('stage2'):
    #         self.get_resnet_conv_block_skip_3(name='conv', filter_size=3, input_channels=64, filters_num=(64,64,256), isAddW2Loss=True)
    #         self.get_resnet_indentity_block_skip_3_param(name='id1', filter_size=3, input_channels=256, filters_num=(64, 64, 256), isAddW2Loss=True)
    #         self.get_resnet_indentity_block_skip_3_param(name='id2', filter_size=3, input_channels=256, filters_num=(64, 64, 256), isAddW2Loss=True)
    #     with tf.variable_scope('stage3'):
    #         self.get_resnet_conv_block_skip_3(name='conv', filter_size=3, input_channels=256, filters_num=(128,128,512), isAddW2Loss=True)
    #         self.get_resnet_indentity_block_skip_3_param(name='id1', filter_size=3, input_channels=512, filters_num=(128, 128, 512), isAddW2Loss=True)
    #         self.get_resnet_indentity_block_skip_3_param(name='id2', filter_size=3, input_channels=512, filters_num=(128, 128, 512), isAddW2Loss=True)
    #         self.get_resnet_indentity_block_skip_3_param(name='id3', filter_size=3, input_channels=512, filters_num=(128, 128, 512), isAddW2Loss=True)
    #     with tf.variable_scope('stage4'):
    #         self.get_resnet_conv_block_skip_3(name='conv', filter_size=3, input_channels=512, filters_num=(256,256,1024), isAddW2Loss=True)
    #         self.get_resnet_indentity_block_skip_3_param(name='id1', filter_size=3, input_channels=1024, filters_num=(256, 256, 1024), isAddW2Loss=True)
    #         self.get_resnet_indentity_block_skip_3_param(name='id2', filter_size=3, input_channels=1024, filters_num=(256, 256, 1024), isAddW2Loss=True)
    #         self.get_resnet_indentity_block_skip_3_param(name='id3', filter_size=3, input_channels=1024, filters_num=(256, 256, 1024), isAddW2Loss=True)
    #         self.get_resnet_indentity_block_skip_3_param(name='id4', filter_size=3, input_channels=1024, filters_num=(256, 256, 1024), isAddW2Loss=True)
    #         self.get_resnet_indentity_block_skip_3_param(name='id5', filter_size=3, input_channels=1024, filters_num=(256, 256, 1024), isAddW2Loss=True)
    #     with tf.variable_scope('stage5'):
    #         self.get_resnet_conv_block_skip_3(name='conv', filter_size=3, input_channels=1024, filters_num=(512,512,2048), isAddW2Loss=True)
    #         self.get_resnet_indentity_block_skip_3_param(name='id1', filter_size=3, input_channels=2048, filters_num=(256, 256, 2048), isAddW2Loss=True)
    #         self.get_resnet_indentity_block_skip_3_param(name='id2', filter_size=3, input_channels=2048, filters_num=(256, 256, 2048), isAddW2Loss=True)
    #         tf.get_variable(name='W1', shape=[1,1,2048,10], dtype=tf.float32)
    #         tf.get_variable(name='b1', shape=[10], dtype=tf.float32)

    def get_resnet_param(self):
        with tf.variable_scope('stage1'):
            self._get_conv_bn_param(filter_size=7, output_channels=64, input_channels=1, isAddW2Loss=True)
        with tf.variable_scope('stage2'):
            self.get_resnet_conv_block_skip_3(name='conv', filter_size=3, input_channels=64, filters_num=(64,64,256), isAddW2Loss=True)
            self.get_resnet_indentity_block_skip_3_param(name='id1', filter_size=3, input_channels=256, filters_num=(32, 32, 256), isAddW2Loss=True)
            self.get_resnet_indentity_block_skip_3_param(name='id2', filter_size=3, input_channels=256, filters_num=(32, 32, 256), isAddW2Loss=True)
        with tf.variable_scope('stage3'):
            self.get_resnet_conv_block_skip_3(name='conv', filter_size=3, input_channels=256, filters_num=(128,128,512), isAddW2Loss=True)
            self.get_resnet_indentity_block_skip_3_param(name='id1', filter_size=3, input_channels=512, filters_num=(128, 128, 512), isAddW2Loss=True)
            self.get_resnet_indentity_block_skip_3_param(name='id2', filter_size=3, input_channels=512, filters_num=(128, 128, 512), isAddW2Loss=True)
            self.get_resnet_indentity_block_skip_3_param(name='id3', filter_size=3, input_channels=512, filters_num=(128, 128, 512), isAddW2Loss=True)
        with tf.variable_scope('stage4'):
            self.get_resnet_conv_block_skip_3(name='conv', filter_size=3, input_channels=512, filters_num=(256,256,1024), isAddW2Loss=True)
            self.get_resnet_indentity_block_skip_3_param(name='id1', filter_size=3, input_channels=1024, filters_num=(256, 256, 1024), isAddW2Loss=True)
            self.get_resnet_indentity_block_skip_3_param(name='id2', filter_size=3, input_channels=1024, filters_num=(256, 256, 1024), isAddW2Loss=True)
            self.get_resnet_indentity_block_skip_3_param(name='id3', filter_size=3, input_channels=1024, filters_num=(256, 256, 1024), isAddW2Loss=True)
            self.get_resnet_indentity_block_skip_3_param(name='id4', filter_size=3, input_channels=1024, filters_num=(256, 256, 1024), isAddW2Loss=True)
            self.get_resnet_indentity_block_skip_3_param(name='id5', filter_size=3, input_channels=1024, filters_num=(256, 256, 1024), isAddW2Loss=True)
        with tf.variable_scope('stage5'):
            self.get_resnet_conv_block_skip_3(name='conv', filter_size=3, input_channels=1024, filters_num=(512,512,2048), isAddW2Loss=True)
            self.get_resnet_indentity_block_skip_3_param(name='id1', filter_size=3, input_channels=2048, filters_num=(256, 256, 2048), isAddW2Loss=True)
            self.get_resnet_indentity_block_skip_3_param(name='id2', filter_size=3, input_channels=2048, filters_num=(256, 256, 2048), isAddW2Loss=True)
            tf.get_variable(name='W1', shape=[1,1,2048,10], dtype=tf.float32)
            tf.get_variable(name='b1', shape=[10], dtype=tf.float32)
        # with tf.variable_scope('stage5'):
        #     self.get_resnet_conv_block_skip_3(name='conv', filter_size=3, input_channels=512, filters_num=(256,256,1024), isAddW2Loss=True)
        #     self.get_resnet_indentity_block_skip_3_param(name='id1', filter_size=3, input_channels=1024, filters_num=(256, 256, 1024), isAddW2Loss=True)
        #     self.get_resnet_indentity_block_skip_3_param(name='id2', filter_size=3, input_channels=1024, filters_num=(256, 256, 1024), isAddW2Loss=True)
        #     tf.get_variable(name='W1', shape=[1,1,1024,10], dtype=tf.float32)
        #     tf.get_variable(name='b1', shape=[10], dtype=tf.float32)

    def conv_block(self, X, isTraining, strides, name):
        with tf.variable_scope(name):
            with tf.variable_scope('a'):
                y = self.conv_bn_relu(X, isTraining, conv_strides=strides, conv_padding='VALID')
            with tf.variable_scope('b'):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='SAME')
            with tf.variable_scope('c'):
                y = self.conv_bn(y, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('d'):
                y2 = self.conv_bn(X, isTraining, conv_strides=strides, conv_padding='VALID')
            y = y+y2
        y = tf.nn.relu(y)
        return y

    def ID_block(self, X, isTraining, name):
        with tf.variable_scope(name):
            with tf.variable_scope('a'):
                y = self.conv_bn_relu(X, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('b'):
                y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='SAME')
            with tf.variable_scope('c'):
                y = self.conv_bn(y, isTraining, conv_strides=1, conv_padding='VALID')            
            y = y+X
            y = tf.nn.relu(y)
        return y

    def resnet_forward(self, X, isTraining):
        with tf.variable_scope('stage1', reuse=True):
            y = self.conv_bn_relu_pool(X, isTraining, conv_strides=1, conv_padding='SAME', pool_size=2, pool_strides=2, pool_padding='VALID')
        with tf.variable_scope('stage2', reuse=True):
            y = self.conv_block(y, isTraining, strides=1, name='conv')
            y = self.ID_block(y, isTraining, name='id1')
            y = self.ID_block(y, isTraining, name='id2')
        with tf.variable_scope('stage3', reuse=True):
            y = self.conv_block(y, isTraining, strides=2, name='conv')
            y = self.ID_block(y, isTraining, name='id1')
            y = self.ID_block(y, isTraining, name='id2')
            y = self.ID_block(y, isTraining, name='id3')
        with tf.variable_scope('stage4', reuse=True):
            y = self.conv_block(y, isTraining, strides=2, name='conv')
            y = self.ID_block(y, isTraining, name='id1')
            y = self.ID_block(y, isTraining, name='id2')
            y = self.ID_block(y, isTraining, name='id3')
            y = self.ID_block(y, isTraining, name='id4')
            y = self.ID_block(y, isTraining, name='id5')
        with tf.variable_scope('stage5', reuse=True):
            y = self.conv_block(y, isTraining, strides=2, name='conv')
            y = self.ID_block(y, isTraining, name='id1')
            y = self.ID_block(y, isTraining, name='id2')
            y = tf.nn.avg_pool(y, ksize=[1,2,2,1],strides=[1,1,1,1],padding='VALID')
            y = tf.nn.conv2d(y, filter=tf.get_variable('W1'), strides=[1,1,1,1], padding='SAME')
            y = tf.reshape(y, shape=[-1, 10])
        # with tf.variable_scope('stage5', reuse=True):
        #     y = self.conv_block(y, isTraining, strides=1, name='conv')
        #     y = self.ID_block(y, isTraining, name='id1')
        #     y = self.ID_block(y, isTraining, name='id2')
        #     y = tf.nn.avg_pool(y, ksize=[1,10,10,1],strides=[1,1,1,1],padding='VALID')
        #     y = tf.nn.conv2d(y, filter=tf.get_variable('W1'), strides=[1,1,1,1], padding='SAME')
        #     y = tf.reshape(y, shape=[-1, 10])
        return y
    
    def loss(self, y_out, y, reg):
        mean_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), y_out)) + reg*tf.add_n(tf.get_collection('loss_W'))
        return mean_loss

    def accuracy(self, y_out, y):
        correct_prediction = tf.equal(tf.cast(tf.argmax(y_out,1), tf.int32), y)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc

    def predict(self, sess, data, X_y_holder):
        batch_size = 128
        isTraining = False
        Z = self.resnet_forward(X_y_holder[0], isTraining)
        # loss = self.loss(Z, X_y_holder[1], reg)
        accuracy = self.accuracy(Z, X_y_holder[1])
        
        X_eval, y_eval = data
        
        m = np.shape(X_eval)[0]
        loss_real_list, accuracy_real_list = [], []
        tm = m//batch_size + 1
        for i in range(tm):
            X_batch = X_eval[i*batch_size:i*batch_size+batch_size,:]
            y_batch = y_eval[i*batch_size:i*batch_size+batch_size]
            feed_dict = {X_y_holder[0]:X_batch, X_y_holder[1]:y_batch}
            accuracy_real_batch = sess.run([accuracy], feed_dict=feed_dict)
            # loss_real_list.append(loss_real_batch)
            accuracy_real_list.append(accuracy_real_batch)
        _, accuracy_real = np.sum(loss_real_list)/tm, np.sum(accuracy_real_list)/tm
        return accuracy_real

    def predict_cnn(self, sess, data, X_y_holder):
        batch_size = 128
        isTraining = False
        Z = self.forward(X_y_holder[0], isTraining)
        # loss = self.loss(Z, X_y_holder[1], reg)
        accuracy = self.accuracy(Z, X_y_holder[1])
        
        X_eval, y_eval = data
        
        m = np.shape(X_eval)[0]
        loss_real_list, accuracy_real_list = [], []
        tm = m//batch_size + 1
        for i in range(tm):
            X_batch = X_eval[i*batch_size:i*batch_size+batch_size,:]
            y_batch = y_eval[i*batch_size:i*batch_size+batch_size]
            feed_dict = {X_y_holder[0]:X_batch, X_y_holder[1]:y_batch}
            accuracy_real_batch = sess.run([accuracy], feed_dict=feed_dict)
            # loss_real_list.append(loss_real_batch)
            accuracy_real_list.append(accuracy_real_batch)
        _, accuracy_real = np.sum(loss_real_list)/tm, np.sum(accuracy_real_list)/tm
        return accuracy_real

    def evaluate(self, sess, data, X_y_holder, loss, accuracy, batch_size):
        X_eval, y_eval = data
        m = np.shape(X_eval)[0]
        loss_real_list, accuracy_real_list = [], []
        tm = m//batch_size + 1
        for i in range(tm):
            X_batch = X_eval[i*batch_size:i*batch_size+batch_size,:]
            y_batch = y_eval[i*batch_size:i*batch_size+batch_size]
            feed_dict = {X_y_holder[0]:X_batch, X_y_holder[1]:y_batch}
            loss_real_batch, accuracy_real_batch = sess.run([loss, accuracy], feed_dict=feed_dict)
            loss_real_list.append(loss_real_batch)
            accuracy_real_list.append(accuracy_real_batch)
        loss_real, accuracy_real = np.sum(loss_real_list)/tm, np.sum(accuracy_real_list)/tm
        return loss_real, accuracy_real

    def train(self, sess, data, X_y_holder, loss, accuracy, optimizer, batch_size):
        X_eval, y_eval = data
        m = np.shape(X_eval)[0]
        loss_real_list, accuracy_real_list = [], []
        tm = m//batch_size + 1
        for i in range(tm):
            X_batch = X_eval[i*batch_size:i*batch_size+batch_size,:]
            y_batch = y_eval[i*batch_size:i*batch_size+batch_size]
            feed_dict = {X_y_holder[0]:X_batch, X_y_holder[1]:y_batch}
            loss_real_batch, accuracy_real_batch, _ = sess.run([loss, accuracy, optimizer], feed_dict=feed_dict)
            loss_real_list.append(loss_real_batch)
            accuracy_real_list.append(accuracy_real_batch)
            
        loss_real, accuracy_real = np.sum(loss_real_list)/tm, np.sum(accuracy_real_list)/tm
        return loss_real, accuracy_real

    def do_it_resnet(self, sess, data, epchos, batch_size, reg, starter_learning_rate, decay_steps, decay_rate):
        X = tf.placeholder(dtype=tf.float32, shape=[None,28,28,1],name='X')
        y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')

        isTraining = True
        Z_train = self.resnet_forward(X, isTraining)
        loss_train = self.loss(Z_train, y, reg)
        accuracy_train = self.accuracy(Z_train, y)

        isTraining = False
        Z_val_test = self.resnet_forward(X, isTraining)
        loss_val_test = self.loss(Z_val_test, y, reg)
        accuracy_val_test = self.accuracy(Z_val_test, y)

        initial_learning_rate = 0.1 * batch_size / 128
        batches_per_epoch = 50000 / batch_size
        self.global_step = tf.train.get_or_create_global_step()

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [500, 6000, 9000, 12000, 15000]
        values = [0.01, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        self.learning_rate = tf.train.piecewise_constant(
            tf.cast(self.global_step, tf.int32), boundaries, values)

        MomentumOptimizer = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate,
            momentum=0.9)
        train_step = MomentumOptimizer.minimize(loss_train, global_step=self.global_step)

        # self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
        #                                         decay_steps, decay_rate, staircase=True, name='learning_rate')
        
        # AdamOptimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # train_step = AdamOptimizer.minimize(loss_train, global_step=self.global_step)

        optimizer = [self.update_op, train_step]

        sess.run(tf.global_variables_initializer())

        X_y_holder = (X, y)
        self.run(sess, data, X_y_holder, loss_train, accuracy_train, loss_val_test, accuracy_val_test, optimizer, batch_size=128, epchos=epchos)        

    def do_it_without_valid(self, sess, data, epchos, batch_size, reg, starter_learning_rate, decay_steps, decay_rate):
        X = tf.placeholder(dtype=tf.float32, shape=[None,22,22,1],name='X')
        y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')
        isTraining = True
        Z_train = self.forward(X, isTraining)
        loss_train = self.loss(Z_train, y, reg)
        accuracy_train = self.accuracy(Z_train, y)

        isTraining = False
        Z_val_test = self.forward(X, isTraining)
        loss_val_test = self.loss(Z_val_test, y, reg)
        accuracy_val_test = self.accuracy(Z_val_test, y)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                decay_steps, decay_rate, staircase=True, name='learning_rate')
        
        AdamOptimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step = AdamOptimizer.minimize(loss_train, global_step=self.global_step)

        optimizer = [self.update_op, train_step]

        sess.run(tf.global_variables_initializer())

        X_y_holder = (X, y)
        self.run_without_valid(sess, data, X_y_holder, loss_train, accuracy_train, loss_val_test, accuracy_val_test, optimizer, batch_size=batch_size, epchos=epchos)        

    def do_it(self, sess, data, epchos, batch_size, reg, starter_learning_rate, decay_steps, decay_rate):
        X = tf.placeholder(dtype=tf.float32, shape=[None,28,28,1],name='X')
        y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')
        isTraining = True
        Z_train = self.forward(X, isTraining)
        loss_train = self.loss(Z_train, y, reg)
        accuracy_train = self.accuracy(Z_train, y)

        isTraining = False
        Z_val_test = self.forward(X, isTraining)
        loss_val_test = self.loss(Z_val_test, y, reg)
        accuracy_val_test = self.accuracy(Z_val_test, y)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                decay_steps, decay_rate, staircase=True, name='learning_rate')
        
        AdamOptimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step = AdamOptimizer.minimize(loss_train, global_step=self.global_step)
        optimizer = [self.update_op, train_step]

        # # Scale the learning rate linearly with the batch size. When the batch size
        # # is 128, the learning rate should be 0.1.
        # initial_learning_rate = 0.1 * batch_size / 128
        # batches_per_epoch = 50000 / batch_size
        # self.global_step = tf.train.get_or_create_global_step()

        # # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        # boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
        # values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
        # self.learning_rate = tf.train.piecewise_constant(
        #     tf.cast(global_step, tf.int32), boundaries, values)

        sess.run(tf.global_variables_initializer())

        X_y_holder = (X, y)
        self.run(sess, data, X_y_holder, loss_train, accuracy_train, loss_val_test, accuracy_val_test, optimizer, batch_size=batch_size, epchos=epchos)        

    def run_without_valid(self, sess, data, X_y_holder, loss_train, accuracy_train, loss_val_test, accuracy_val_test, optimizer, batch_size, epchos):
        """X_y_holder: (X, y)
        data = (X_train, y_train, X_valid, y_valid, X_test, y_test)
        """
        X_train, y_train, X_test, y_test = data
        data_train = (X_train, y_train)
        data_test = (X_test, y_test)
        
        loss_real_train, accuracy_real_train = self.evaluate(sess, data_train, X_y_holder, loss_val_test, accuracy_val_test, batch_size)
        loss_real_test, accuracy_real_test = self.evaluate(sess, data_test, X_y_holder, loss_val_test, accuracy_val_test, batch_size)

        self.train_loss_log.append(loss_real_train)
        self.train_accuracy_log.append(accuracy_real_train)
        self.test_loss_log.append(loss_real_test)
        self.test_accuracy_log.append(accuracy_real_test)
        print('epoch %d Loss: train %0.5f, test %0.5f, acc: train %0.5f, test %0.5f'%(
                0, loss_real_train, loss_real_test, accuracy_real_train, accuracy_real_test))

        for epcho in range(epchos):
            X_train, y_train = data_train
            idx = np.random.permutation(X_train.shape[0])
            X_train_tmp = X_train[idx, :]
            y_train_tmp = y_train[idx]
            data_train = (X_train_tmp, y_train_tmp)
            loss_real_train, accuracy_real_train = self.train(sess, data_train, X_y_holder, loss_train, accuracy_train, optimizer, batch_size)
            loss_real_test, accuracy_real_test = self.evaluate(sess, data_test, X_y_holder, loss_val_test, accuracy_val_test, batch_size)
            self.train_loss_log.append(loss_real_train)
            self.train_accuracy_log.append(accuracy_real_train)
            self.test_loss_log.append(loss_real_test)
            self.test_accuracy_log.append(accuracy_real_test)

            print('epoch %d Loss: train %0.5f, test %0.5f, acc: train %0.5f, test %0.5f'%(
                    epcho+1, loss_real_train, loss_real_test, accuracy_real_train, accuracy_real_test))
            print('global step %d, learning rate %f'%(sess.run(self.global_step), sess.run(self.learning_rate)))

    def run(self, sess, data, X_y_holder, loss_train, accuracy_train, loss_val_test, accuracy_val_test, optimizer, batch_size, epchos):
        """X_y_holder: (X, y)
        data = (X_train, y_train, X_valid, y_valid, X_test, y_test)
        """
        X_train, y_train, X_valid, y_valid, X_test, y_test = data
        data_train = (X_train, y_train)
        data_valid = (X_valid, y_valid)
        data_test = (X_test, y_test)
        
        loss_real_train, accuracy_real_train = self.evaluate(sess, data_train, X_y_holder, loss_val_test, accuracy_val_test, batch_size)
        loss_real_valid, accuracy_real_valid = self.evaluate(sess, data_valid, X_y_holder, loss_val_test, accuracy_val_test, batch_size)
        loss_real_test, accuracy_real_test = self.evaluate(sess, data_test, X_y_holder, loss_val_test, accuracy_val_test, batch_size)

        self.train_loss_log.append(loss_real_train)
        self.train_accuracy_log.append(accuracy_real_train)
        self.test_loss_log.append(loss_real_test)
        self.test_accuracy_log.append(accuracy_real_test)
        self.valid_loss_log.append(loss_real_valid)
        self.valid_accuracy_log.append(accuracy_real_valid)
        print('epoch %d Loss: train %0.5f, valid %0.5f, test %0.5f, acc: train %0.5f, valid %0.5f, test %0.5f'%(
                0, loss_real_train, loss_real_valid, loss_real_test, accuracy_real_train, accuracy_real_valid, accuracy_real_test))

        for epcho in range(epchos):
            X_train, y_train = data_train
            idx = np.random.permutation(X_train.shape[0])
            X_train_tmp = X_train[idx, :]
            y_train_tmp = y_train[idx]
            data_train = (X_train_tmp, y_train_tmp)
            loss_real_train, accuracy_real_train = self.train(sess, data_train, X_y_holder, loss_train, accuracy_train, optimizer, batch_size)
            loss_real_valid, accuracy_real_valid = self.evaluate(sess, data_valid, X_y_holder, loss_val_test, accuracy_val_test, batch_size)
            loss_real_test, accuracy_real_test = self.evaluate(sess, data_test, X_y_holder, loss_val_test, accuracy_val_test, batch_size)
            self.train_loss_log.append(loss_real_train)
            self.train_accuracy_log.append(accuracy_real_train)
            self.test_loss_log.append(loss_real_test)
            self.test_accuracy_log.append(accuracy_real_test)
            self.valid_loss_log.append(loss_real_valid)
            self.valid_accuracy_log.append(accuracy_real_valid)
            
            with open('min_loss1.txt', 'r+') as fp:
                line = fp.readline()
                tmp_acc = float(line)
                if accuracy_real_test > tmp_acc:
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, "model.ckpt")
                    print("Model saved in file: %s" % save_path)
                    fp.seek(0,0)
                    fp.truncate()
                    fp.write(str(accuracy_real_test))

            print('epoch %d Loss: train %0.5f, valid %0.5f, test %0.5f, acc: train %0.5f, valid %0.5f, test %0.5f'%(
                    epcho+1, loss_real_train, loss_real_valid, loss_real_test, accuracy_real_train, accuracy_real_valid, accuracy_real_test))
            print('global step %d, learning rate %f'%(sess.run(self.global_step), sess.run(self.learning_rate)))

    def show_log(self, style='-o'):
        epcho = len(self.train_loss_log)
        x = list(range(epcho))
        plt.plot(x, self.train_loss_log, style)
        plt.plot(x, self.valid_loss_log, style)
        plt.plot(x, self.test_loss_log, style)
        plt.legend(['train', 'valid', 'test'])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('loss')
        plt.show()
        plt.plot(x, self.train_accuracy_log, style)
        plt.plot(x, self.valid_accuracy_log, style)
        plt.plot(x, self.test_accuracy_log, style)
        plt.legend(['train', 'valid', 'test'])
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.title('accuracy')
        plt.show()

# class cnn_model:
#     def __init__(self):
#         """init
#         """
#         self.struct = {
#             'conv1':{
#                 'filter_num':64,
#                 'kernel_size':3,
#                 'channel':3,
#                 'random_seed':1
#             },
#             'conv2':{
#                 'filter_num':64,
#                 'kernel_size':3,
#                 'channel':64,
#                 'random_seed':2
#             },
#             'conv3':{
#                 'filter_num':32,
#                 'kernel_size':3,
#                 'channel':64,
#                 'random_seed':3
#             },
#             'conv4':{
#                 'filter_num':10,
#                 'kernel_size':4,
#                 'channel':32,
#                 'random_seed':3                
#             }
#         }
#         self.layer_num = len(self.struct.items())
#         self.class_out = self.struct['conv'+str(len(self.struct.items()))]['filter_num']

#         self.conv = {}
#         self.hyper_params = {}

#         self.train_X, self.train_y = None, None
#         self.valid_y, self.valid_y = None, None
#         self.test_X, self.test_y = None, None
        
#         self.loss_log = {
#             'train':[],
#             'valid':[],
#             'test':[],
#             'train_loss':[],
#             'valid_loss':[],
#             'test_loss':[],
            
#             'train_acc':[],
#             'valid_acc':[],
#             'test_acc':[]
#         }        
#         self._get_param()


#     def _get_conv_param(self, filter_num=32, channels=3, filter_size=3, stddev=0.001, dtype=tf.float32, seed=0):
#         """get conv param
#         """
#         tf.set_random_seed(seed)
#         W = tf.Variable(tf.random_normal(shape=[filter_size, filter_size, channels, filter_num],
#                             mean=0.0,
#                             stddev=stddev,
#                             dtype=dtype), 
#                         name='W')
#         b = tf.Variable(tf.zeros(filter_num, dtype=dtype), 
#                         name='b')
#         return W, b

#     def _get_conv_bn_param(self, filter_num=32, channels=3, filter_size=3, stddev=0.001, dtype=tf.float32, seed=0):
#         """get conv bn param
#         """
#         tf.set_random_seed(seed)
#         W, b = self._get_conv_param(filter_num, channels, filter_size, stddev, dtype)
#         bn_scale = tf.Variable(tf.ones(filter_num, dtype=dtype), name='bn_scale')
#         bn_offset = tf.Variable(tf.zeros(filter_num), dtype=dtype, name='bn_offset')
#         bn_running_mean = tf.Variable(tf.zeros(filter_num, dtype=dtype), name='bn_running_mean')
#         bn_running_var = tf.Variable(tf.ones(filter_num), dtype=dtype, name='bn_running_var')
#         return W, b, bn_scale, bn_offset, bn_running_mean, bn_running_var

#     def bacth_normalization(self, conv, l, offset, scale, isTraining=True, eps=1e-8):
#         if isTraining:
#             mu, var = tf.nn.moments(conv, axes=[0,1,2], keep_dims=True)
#             self.conv['conv'+str(l)]['bn_running_mean'] = self.conv['momentum']*mu + (1-self.conv['momentum'])*self.conv['conv'+str(l)]['bn_running_mean']
#             self.conv['conv'+str(l)]['bn_running_var'] = self.conv['momentum']*var + (1-self.conv['momentum'])*self.conv['conv'+str(l)]['bn_running_var']
#         else:
#             mu, var = self.conv['conv'+str(l)]['bn_running_mean'], self.conv['conv'+str(l)]['bn_running_var']
#         return tf.nn.batch_normalization(conv, mu, var, offset, scale, eps)
        
#     def _get_param(self):
#         """get params
#         """
#         for l in range(1, self.layer_num):
#             self.conv['conv'+str(l)] = {}
#             with tf.variable_scope('conv'+str(l)):
#                 self.conv['conv'+str(l)]['W'], \
#                 self.conv['conv'+str(l)]['b'], \
#                 self.conv['conv'+str(l)]['bn_scale'], \
#                 self.conv['conv'+str(l)]['bn_offset'],\
#                 self.conv['conv'+str(l)]['bn_running_mean'], \
#                 self.conv['conv'+str(l)]['bn_running_var'] = self._get_conv_bn_param(self.struct['conv'+str(l)]['filter_num'],
#                                                                                 self.struct['conv'+str(l)]['channel'],
#                                                                                 self.struct['conv'+str(l)]['kernel_size'],
#                                                                                 self.struct['conv'+str(l)]['random_seed'])
#         l = self.layer_num
#         self.conv['conv'+str(l)] = {}
#         with tf.variable_scope('conv'+str(l)):
#             self.conv['conv'+str(l)]['W'], \
#             self.conv['conv'+str(l)]['b'] = self._get_conv_param(self.struct['conv'+str(l)]['filter_num'],
#                                                                 self.struct['conv'+str(l)]['channel'],
#                                                                 self.struct['conv'+str(l)]['kernel_size'],
#                                                                 self.struct['conv'+str(l)]['random_seed'])
#         self.conv['momentum'] = tf.constant(0.9, dtype=tf.float32)

#     def forward(self, X, isTraining=True):
#         last_layer = X
#         for l in range(1, self.layer_num):
#             conv = tf.nn.conv2d(last_layer, self.conv['conv'+str(l)]['W'], strides=[1,1,1,1], padding='SAME')
#             bn = self.bacth_normalization(conv, l, self.conv['conv'+str(l)]['bn_offset'], self.conv['conv'+str(l)]['bn_scale'], isTraining)
#             relu = tf.nn.relu(bn)
#             last_layer = tf.nn.max_pool(relu, [1,3,3,1], strides=[1,2,2,1], padding='SAME')
#         l = self.layer_num
#         z = tf.nn.conv2d(last_layer, self.conv['conv'+str(l)]['W'], strides=[1,1,1,1], padding='VALID')
#         z = tf.reshape(z, shape=[-1,self.class_out])
#         return z

#     def loss(self, y, z, lamb):
#         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=z))
#         for l in range(1, self.layer_num+1):
#             loss += lamb*tf.reduce_sum(self.conv['conv'+str(l)]['W']**2)
#         return loss
    
#     def accuracy(self, y, z):
#         # z = self.forward(X, isTraining=False)
#         correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(y,1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         return accuracy

#     def predict(self, X):
#         z = self.forward(X, isTraining=False)
#         y_pred = tf.nn.softmax(z)
#         return y_pred

#     def plot_log(self):
#         plt.plot(self.loss_log['train'], '-o')
#         plt.plot(self.loss_log['valid'], '-o')
#         plt.plot(self.loss_log['test'], '-o')
#         plt.legend(['train', 'valid', 'test'])
#         plt.xlabel('epochs')
#         plt.ylabel('loss')
#         plt.show()





#     def do_it(self, sess, data, name, batch_size=64, epchos=1, learning_rate=1e-3, regular = 0, isRequireAcc = True, dtype=tf.float32):
#         """name require :'train', 'valid' or 'test'
#         """
#         if name == 'train':
#             isTraining = True
#         elif name == 'valid' or name == 'test':
#             isTraining = False
#             epchos = 1
#         X_data, y_data = data        
#         assert X_data.shape[0] == y_data.shape[0]
#         X = tf.placeholder(dtype=dtype, shape=[None, 32, 32, 3])
#         y = tf.placeholder(dtype=tf.int64, shape=[None])

#         y_hot = tf.one_hot(y, self.class_out)
#         sess.run(y_hot, feed_dict={X:X_data, y:y_data})

#         z = self.forward(X, isTraining=isTraining)
#         loss = self.loss(y_hot,z,regular)
#         if isRequireAcc:
#             acc = self.accuracy(y_hot,z)
#         optimizer = tf.train.AdamOptimizer(learning_rate)
#         train_step = optimizer.minimize(loss)

#         tf.global_variables_initializer().run()

#         for i in range(epchos):
#             index = np.arange(np.shape(X_data)[0])
#             loss_batch_list = []
#             acc_batch_list = []
#             for j in range(int(np.ceil(np.shape(X_data)[0]/batch_size))):
#                 idx = index[j*batch_size:j*batch_size+batch_size]
#                 batch_X = X_data[idx, :]
#                 batch_y = y_data[idx]
#                 feed_dict={X:batch_X, y:batch_y}
#                 if isRequireAcc:
#                     loss_batch, acc_batch, _ = sess.run([loss, acc, train_step], feed_dict=feed_dict)
#                     acc_batch_list.append(acc_batch)
#                 else:
#                     loss_batch, _ = sess.run([loss, train_step], feed_dict=feed_dict)
#                 loss_batch_list.append(loss_batch)
            
#             loss_real = np.sum(loss_batch_list)/len(loss_batch_list)
#             self.loss_log[name+'_loss'].append(loss_real)
#             if isRequireAcc:
#                 acc_real = np.sum(acc_batch_list)/len(acc_batch_list)
#                 self.loss_log[name+'_acc'].append(acc_real)
#             if isRequireAcc:
#                 print(('epcho %d '+name+' loss: %0.6f, acc: %0.6f')%(i, loss_real, acc_real))
#             else:
#                 print(('epcho %d '+name+' loss: %0.6f')%(i, loss_real))

#     def train(self, sess, train_X, train_y, val_X, val_y, test_X, test_y, epochs = 10, learning_rate = 0.001, batch_size = 64, regular = 0.001, dtype=tf.float32):
#         m, w, h, c = train_X.shape
#         # clear param first
#         # self._get_param()
#         #
#         X = tf.placeholder(dtype=dtype, shape=[None, w, h, c])
#         y = tf.placeholder(dtype=tf.int64, shape=[None])
#         z = self.forward(X, isTraining=True)
#         loss = self.loss(y, z, regular)
#         optimizer = tf.train.AdamOptimizer(learning_rate)
#         train_step = optimizer.minimize(loss)
#         loss_test_op = self.loss(y, self.forward(X, isTraining=False), regular)
#         acc_op = self.accuracy(y, X)

#         variables = [loss, train_step]
#         #
#         sess.run(tf.global_variables_initializer())
    
#         index = np.arange(m)
#         for i in range(epochs):
#             for j in range(int(np.ceil(m/batch_size))):
#                 idx = index[j*batch_size:j*batch_size+batch_size]
#                 batch_X = train_X[idx, :]
#                 batch_y = train_y[idx]
#                 feed_dict={X:batch_X, y:batch_y}
#                 loss_batch, _ = sess.run(variables, feed_dict=feed_dict)
#                 self.loss_log['train_losses'].append(loss_batch)
#             loss_train = np.sum(self.loss_log['train_losses'])/len(self.loss_log['train_losses'])
#             self.loss_log['train'].append(loss_train)

#             index = np.arange(np.shape(val_X)[0])
#             for j in range(int(np.ceil(np.shape(val_X)[0]/batch_size))):
#                 idx = index[j*batch_size:j*batch_size+batch_size]
#                 print(idx.shape)
#                 batch_X = val_X[idx, :]            
#                 batch_y = val_y[idx]
#                 feed_dict={X:batch_X, y:batch_y}
#                 acc_batch = sess.run(acc_op, feed_dict=feed_dict)
#                 self.loss_log['val_acc'].append(acc_batch)
#             loss_val = np.sum(self.loss_log['val_losses'])/len(self.loss_log['val_losses'])
#             self.loss_log['valid'].append(loss_val)
#             self.loss_log['val_acc'].append()

#             index = np.arange(np.shape(test_X)[0])
#             for j in range(int(np.ceil(np.shape(test_X)[0]/batch_size))):
#                 idx = index[j*batch_size:j*batch_size+batch_size]
#                 batch_X = test_X[idx, :]
#                 batch_y = test_y[idx]
#                 feed_dict={X:batch_X, y:batch_y}
#                 loss_batch = sess.run(loss_test_op, feed_dict=feed_dict)
#                 self.loss_log['test_losses'].append(loss_batch)
#             loss_test = np.sum(self.loss_log['test_losses'])/len(self.loss_log['test_losses'])
#             self.loss_log['test'].append(loss_test)


#         # index = np.arange(np.shape(val_X)[0])
#         # for j in range(int(np.ceil(np.shape(val_X)[0]/batch_size))):
#         #     idx = index[j*batch_size:j*batch_size+batch_size]
#         #     print(idx.shape)
#         #     batch_X = val_X[idx, :]            
#         #     batch_y = val_y[idx]
#         #     feed_dict={X:batch_X, y:batch_y}
#         #     loss_batch = sess.run(loss_test_op, feed_dict=feed_dict)
#         #     self.loss_log['val_losses'].append(loss_batch)
#         # loss_val = np.sum(self.loss_log['val_losses'])/len(self.loss_log['val_losses'])
#         # self.loss_log['valid'].append(loss_val)
        
#         # index = np.arange(np.shape(test_X)[0])
#         # for j in range(int(np.ceil(np.shape(test_X)[0]/batch_size))):
#         #     idx = index[j*batch_size:j*batch_size+batch_size]
#         #     batch_X = test_X[idx, :]
#         #     batch_y = test_y[idx]
#         #     feed_dict={X:batch_X, y:batch_y}
#         #     loss_batch = sess.run(loss_test_op, feed_dict=feed_dict)
#         #     self.loss_log['test_losses'].append(loss_batch)
#         # loss_test = np.sum(self.loss_log['test_losses'])/len(self.loss_log['test_losses'])
#         # self.loss_log['test'].append(loss_test)
        
#         print('epcho %d:train loss:%0.6f, valid loss:%0.6f, test loss%0.6f'%(i, loss_train, loss_val, loss_test))
#     def train(self, sess, train_X, train_y, val_X, val_y, test_X, test_y, epochs = 10, learning_rate = 0.001, batch_size = 64, regular = 0.001, dtype=tf.float32):
#         m, w, h, c = train_X.shape
#         # clear param first
#         # self._get_param()
#         #
#         X = tf.placeholder(dtype=dtype, shape=[None, w, h, c])
#         y = tf.placeholder(dtype=tf.int64, shape=[None])
#         z = self.forward(X, isTraining=True)
#         loss = self.loss(y, z, regular)
#         optimizer = tf.train.AdamOptimizer(learning_rate)
#         train_step = optimizer.minimize(loss)
#         loss_test_op = self.loss(y, self.forward(X, isTraining=False), regular)
#         acc_op = self.accuracy(y, X)

#         variables = [loss, train_step]
#         #
#         sess.run(tf.global_variables_initializer())
    
#         index = np.arange(m)
#         for i in range(epochs):
#             for j in range(int(np.ceil(m/batch_size))):
#                 idx = index[j*batch_size:j*batch_size+batch_size]
#                 batch_X = train_X[idx, :]
#                 batch_y = train_y[idx]
#                 feed_dict={X:batch_X, y:batch_y}
#                 loss_batch, _ = sess.run(variables, feed_dict=feed_dict)
#                 self.loss_log['train_losses'].append(loss_batch)
#             loss_train = np.sum(self.loss_log['train_losses'])/len(self.loss_log['train_losses'])
#             self.loss_log['train'].append(loss_train)

#             index = np.arange(np.shape(val_X)[0])
#             for j in range(int(np.ceil(np.shape(val_X)[0]/batch_size))):
#                 idx = index[j*batch_size:j*batch_size+batch_size]
#                 print(idx.shape)
#                 batch_X = val_X[idx, :]            
#                 batch_y = val_y[idx]
#                 feed_dict={X:batch_X, y:batch_y}
#                 acc_batch = sess.run(acc_op, feed_dict=feed_dict)
#                 self.loss_log['val_acc'].append(acc_batch)
#             loss_val = np.sum(self.loss_log['val_losses'])/len(self.loss_log['val_losses'])
#             self.loss_log['valid'].append(loss_val)
#             self.loss_log['val_acc'].append()

#             index = np.arange(np.shape(test_X)[0])
#             for j in range(int(np.ceil(np.shape(test_X)[0]/batch_size))):
#                 idx = index[j*batch_size:j*batch_size+batch_size]
#                 batch_X = test_X[idx, :]
#                 batch_y = test_y[idx]
#                 feed_dict={X:batch_X, y:batch_y}
#                 loss_batch = sess.run(loss_test_op, feed_dict=feed_dict)
#                 self.loss_log['test_losses'].append(loss_batch)
#             loss_test = np.sum(self.loss_log['test_losses'])/len(self.loss_log['test_losses'])
#             self.loss_log['test'].append(loss_test)


#         # index = np.arange(np.shape(val_X)[0])
#         # for j in range(int(np.ceil(np.shape(val_X)[0]/batch_size))):
#         #     idx = index[j*batch_size:j*batch_size+batch_size]
#         #     print(idx.shape)
#         #     batch_X = val_X[idx, :]            
#         #     batch_y = val_y[idx]
#         #     feed_dict={X:batch_X, y:batch_y}
#         #     loss_batch = sess.run(loss_test_op, feed_dict=feed_dict)
#         #     self.loss_log['val_losses'].append(loss_batch)
#         # loss_val = np.sum(self.loss_log['val_losses'])/len(self.loss_log['val_losses'])
#         # self.loss_log['valid'].append(loss_val)
        
#         # index = np.arange(np.shape(test_X)[0])
#         # for j in range(int(np.ceil(np.shape(test_X)[0]/batch_size))):
#         #     idx = index[j*batch_size:j*batch_size+batch_size]
#         #     batch_X = test_X[idx, :]
#         #     batch_y = test_y[idx]
#         #     feed_dict={X:batch_X, y:batch_y}
#         #     loss_batch = sess.run(loss_test_op, feed_dict=feed_dict)
#         #     self.loss_log['test_losses'].append(loss_batch)
#         # loss_test = np.sum(self.loss_log['test_losses'])/len(self.loss_log['test_losses'])
#         # self.loss_log['test'].append(loss_test)
        
#         print('epcho %d:train loss:%0.6f, valid loss:%0.6f, test loss%0.6f'%(i, loss_train, loss_val, loss_test))



#     conv2_conv = tf.nn.conv2d(conv1_pool, conv2_W, strides=[1,1,1,1], padding='SAME', data_format='NHWC') + conv2_b
#     conv2_bn = tf.layers.batch_normalization(conv2_conv, training=is_training)
#     conv2_activ = tf.nn.relu(conv2_bn)
#     conv2_pool = tf.nn.max_pool(conv2_activ, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC')

#     conv3_conv = tf.nn.conv2d(conv2_pool, conv3_W, strides=[1,1,1,1], padding='SAME', data_format='NHWC') + conv3_b
#     conv3_bn = tf.layers.batch_normalization(conv3_conv, training=is_training)
#     conv3_activ = tf.nn.relu(conv3_bn)
#     conv3_pool = tf.nn.max_pool(conv3_activ, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC')

#     conv4_conv = tf.nn.conv2d(conv3_pool, conv4_W, strides=[1,1,1,1], padding='VALID', data_format='NHWC') + conv4_b



# y_train_argu = np.append(y_train, y_mirror)
# y_train_argu = np.append(y_train_argu, y_flip)
# y_train_argu = np.append(y_train_argu, y_trans1)
# y_train_argu = np.append(y_train_argu, y_trans2)
# y_train_argu = np.append(y_train_argu, y_trans3)
# y_train_argu = np.append(y_train_argu, y_trans4)
# y_train_argu = np.append(y_train_argu, y_light)
# y_train_argu = np.append(y_train_argu, y_dark)
# y_train_argu = np.append(y_train_argu, y_light_plus)
# y_train_argu = np.append(y_train_argu, y_dark_plus)