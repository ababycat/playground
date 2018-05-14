#!usr/bin/env python3

"""ResNet Model"""

import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class ResNet_Model:
    """ResNet Model"""
    def __init__(self,
                 momentum=0.997,
                 batch_size=64,
                 seed=0,
                 category=10,
                 dtype=tf.float32,
                 X_shape=[None, 28, 28, 1], y_shape=[None],
                 reg=0,
                 starter_learning_rate=1e-2,
                 decay_steps=800,
                 decay_rate=0.5,
                 epchos=1):

        np.random.seed = seed
        self.dtype = dtype

        self.category = category
        self.momentum = momentum
        self.global_step = None
        self.learning_rate = None
        self.reg = reg
        self.seed = seed
        self.batch_size = batch_size
        self.starter_learning_rate = starter_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.epchos = epchos
        
        self.epcho_now = 0

        self.update_op = []

        self.X = tf.placeholder(self.dtype, X_shape, 'X')
        self.y = tf.placeholder(self.dtype, y_shape, 'y')

        self.op_y = None
        self.op_loss = None
        self.op_acc = None

        self.op_y_eval = None
        self.op_loss_eval = None
        self.op_acc_eval = None

        self.log_idex = []
        self.log_loss = []
        self.log_acc = []
        self.log_learning_rate = []

        self.log_idex_eval = []
        self.log_loss_eval = []
        self.log_acc_eval = []

        self.config = {}
        self.log = {
            'train_loss':{'x':[], 'y':[]},
            'train_acc':{'x':[], 'y':[]},
            'learning_rate':{'x':[], 'y':[]},
            'train_loss_epcho':{'x':[], 'y':[]},
            'valid_loss_epcho':{'x':[], 'y':[]},
            'train_acc_epcho':{'x':[], 'y':[]},
            'valid_acc_epcho':{'x':[], 'y':[]},
            'epcho':{'x':[], 'y':[]}
        }

    def show_log(self, style='-o'):
        """show log"""
        plt.plot(self.log['train_loss']['x'], self.log['train_loss']['y'])
        plt.plot(self.log['train_loss_epcho']['x'], self.log['train_loss_epcho']['y'], style)
        plt.plot(self.log['valid_loss_epcho']['x'], self.log['valid_loss_epcho']['y'], style)
        plt.legend(['train loss step', 'train loss epcho', 'valid loss epcho'])
        plt.title('Loss')
        plt.show()
        plt.plot(self.log['train_acc']['x'], self.log['train_acc']['y'])
        plt.plot(self.log['train_acc_epcho']['x'], self.log['train_acc_epcho']['y'], style)
        plt.plot(self.log['valid_acc_epcho']['x'], self.log['valid_acc_epcho']['y'], style)
        plt.legend(['train acc step', 'train acc epcho', 'valid acc epcho'])
        plt.title('Accuracy')
        plt.show()

    def _log_append(self, name, x, y):
        self.log[name]['x'].append(x)
        self.log[name]['y'].append(y)

    def log_append(self, sess, train_loss, train_acc):
        x = sess.run(self.global_step)
        self._log_append('train_loss', x, train_loss)
        self._log_append('train_acc', x, train_acc)
        self._log_append('learning_rate', x, self.learning_rate)

    def log_append_to_epcho(self, sess, train_loss, train_acc, valid_loss, valid_acc):
        x = sess.run(self.global_step)
        self._log_append('train_loss_epcho', x, train_loss)
        self._log_append('train_acc_epcho', x, train_acc)
        self._log_append('valid_loss_epcho', x, valid_loss)
        self._log_append('valid_acc_epcho', x, valid_acc)
        self._log_append('epcho', x, self.epcho_now)

    def get_graph(self):
        """Get Graph"""
        self.op_y = self._op_forward(isTraining=True)
        loss = tf.losses.softmax_cross_entropy(tf.one_hot(tf.cast(self.y, tf.int32), self.category), self.op_y)
        self.op_loss = tf.reduce_mean(loss) + self.reg * tf.add_n(tf.get_collection('loss_W'))
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.op_y, 1), tf.int32), tf.cast(self.y, tf.int32))
        self.op_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.op_y_eval = self._op_forward(isTraining=False)
        loss = tf.losses.softmax_cross_entropy(tf.one_hot(tf.cast(self.y, tf.int32), self.category), self.op_y_eval)
        self.op_loss_eval = tf.reduce_mean(loss) + self.reg * tf.add_n(tf.get_collection('loss_W'))
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.op_y_eval, 1), tf.int32), tf.cast(self.y, tf.int32))
        self.op_acc_eval = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                        self.decay_steps, self.decay_rate, staircase=True, name='learning_rate')
        AdamOptimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step = AdamOptimizer.minimize(self.op_loss, global_step=self.global_step)
        self.optimizer = [self.update_op, train_step]

    def get_variable(self):
        with tf.variable_scope('stage1'):
            self._var_get_conv_bn(7, 64, 1, True)
        with tf.variable_scope('stage2'):
            self._var_get_resnet_conv_block_skip_3(
                'conv', 3, 64, (64, 64, 256), True)
            self._var_get_resnet_indentity_block_skip_3(
                'id1', 3, 256, (32, 32, 256), True)
            self._var_get_resnet_indentity_block_skip_3(
                'id2', 3, 256, (32, 32, 256), True)
        with tf.variable_scope('stage3'):
            self._var_get_resnet_conv_block_skip_3(
                'conv', 3, 256, (128, 128, 512), True)
            self._var_get_resnet_indentity_block_skip_3(
                'id1', 3, 512, (128, 128, 512), True)
            self._var_get_resnet_indentity_block_skip_3(
                'id2', 3, 512, (128, 128, 512), True)
            self._var_get_resnet_indentity_block_skip_3(
                'id3', 3, 512, (128, 128, 512), True)
        with tf.variable_scope('stage4'):
            self._var_get_resnet_conv_block_skip_3(
                'conv', 3, 512, (256, 256, 1024), True)
            self._var_get_resnet_indentity_block_skip_3(
                'id1', 3, 1024, (256, 256, 1024), True)
            self._var_get_resnet_indentity_block_skip_3(
                'id2', 3, 1024, (256, 256, 1024), True)
            self._var_get_resnet_indentity_block_skip_3(
                'id3', 3, 1024, (256, 256, 1024), True)
            self._var_get_resnet_indentity_block_skip_3(
                'id4', 3, 1024, (256, 256, 1024), True)
            self._var_get_resnet_indentity_block_skip_3(
                'id5', 3, 1024, (256, 256, 1024), True)
        with tf.variable_scope('stage5'):
            self._var_get_resnet_conv_block_skip_3(
                'conv', 3, 1024, (512, 512, 2048), True)
            self._var_get_resnet_indentity_block_skip_3(
                'id1', 3, 2048, (256, 256, 2048), True)
            self._var_get_resnet_indentity_block_skip_3(
                'id2', 3, 2048, (256, 256, 2048), True)
            self._var_get_conv(2, 10, 2048, 0.001, True)

    def evaluate(self, sess, data):
        bs = self.batch_size
        X_eval, y_eval = data
        m = np.shape(X_eval)[0]
        loss_real_list, accuracy_real_list = [], []
        tm = m // bs + 1
        for i in range(tm):
            X_batch = X_eval[i * bs:i * bs + bs, :]
            y_batch = y_eval[i * bs:i * bs + bs]
            feed_dict = {self.X: X_batch, self.y: y_batch}
            loss_real_batch, accuracy_real_batch = sess.run([self.op_loss_eval, self.op_acc_eval], feed_dict=feed_dict)
            loss_real_list.append(loss_real_batch)
            accuracy_real_list.append(accuracy_real_batch)
        loss_real, accuracy_real = np.sum(loss_real_list)/tm, np.sum(accuracy_real_list)/tm
        return loss_real, accuracy_real

    def train(self, sess, data):
        bs = self.batch_size
        X_eval, y_eval = data
        m = np.shape(X_eval)[0]
        loss_real_list, accuracy_real_list = [], []
        tm = m // bs
        for i in range(tm):
            X_batch = X_eval[i*bs:i*bs+bs, :]
            y_batch = y_eval[i*bs:i*bs+bs]
            feed_dict = {self.X: X_batch, self.y: y_batch}
            loss_real_batch, accuracy_real_batch, _ = sess.run(
                [self.op_loss, self.op_acc, self.optimizer], feed_dict=feed_dict)
            loss_real_list.append(loss_real_batch)
            accuracy_real_list.append(accuracy_real_batch)
            self.log_append(sess, loss_real_batch, accuracy_real_batch)
        loss_real, accuracy_real = np.sum(loss_real_list)/tm, np.sum(accuracy_real_list)/tm
        return loss_real, accuracy_real

    def run(self, sess, data):
        """X_y_holder: (X, y)
        data = (X_train, y_train, X_valid, y_valid, X_test, y_test)
        """
        X_train, y_train, X_valid, y_valid, X_test, y_test = data
        data_train = (X_train, y_train)
        data_valid = (X_valid, y_valid)
        data_test = (X_test, y_test)

        loss_real_train, accuracy_real_train = self.evaluate(sess, data_train)
        loss_real_valid, accuracy_real_valid = self.evaluate(sess, data_valid)
        self.log_append_to_epcho(sess, loss_real_train, accuracy_real_train, loss_real_valid, accuracy_real_valid)
        self.log_append(sess, loss_real_train, accuracy_real_train)

        print('epoch %d Loss: train %0.5f, valid %0.5f, acc: train %0.5f, valid %0.5f' % (
            0, loss_real_train, loss_real_valid, accuracy_real_train, accuracy_real_valid))

        for epcho in range(1,self.epchos+1):
            np.random.seed = self.seed + 1
            X_train, y_train = data_train
            idx = np.random.permutation(X_train.shape[0])
            X_train_tmp = X_train[idx, :]
            y_train_tmp = y_train[idx]
            data_train = (X_train_tmp, y_train_tmp)
            loss_real_train, accuracy_real_train = self.train(sess, data_train)
            loss_real_valid, accuracy_real_valid = self.evaluate(sess, data_valid)
            self.log_append_to_epcho(sess, loss_real_train, accuracy_real_train, loss_real_valid, accuracy_real_valid)

            with open('min_loss1.txt', 'r+') as fp:
                line = fp.readline()
                tmp_acc = float(line)
                if accuracy_real_valid > tmp_acc:
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, "model.ckpt")
                    print("Model saved in file: %s" % save_path)
                    fp.seek(0, 0)
                    fp.truncate()
                    fp.write(str(accuracy_real_valid))

            print('epoch %d Loss: train %0.5f, valid %0.5f, acc: train %0.5f, valid %0.5f' % (
                epcho, loss_real_train, loss_real_valid, accuracy_real_train, accuracy_real_valid))
            print('global step %d, learning rate %f' %
                  (sess.run(self.global_step), sess.run(self.learning_rate)))

    def _op_forward(self, isTraining):
        with tf.variable_scope('stage1', reuse=True):
            y = self._op_conv_bn_relu_pool(self.X, isTraining, conv_strides=1, conv_padding='SAME',
                                           pool_size=2, pool_strides=2, pool_padding='VALID')
        with tf.variable_scope('stage2', reuse=True):
            y = self._op_conv_block(y, isTraining, strides=1, name='conv')
            y = self._op_ID_block(y, isTraining, name='id1')
            y = self._op_ID_block(y, isTraining, name='id2')
        with tf.variable_scope('stage3', reuse=True):
            y = self._op_conv_block(y, isTraining, strides=2, name='conv')
            y = self._op_ID_block(y, isTraining, name='id1')
            y = self._op_ID_block(y, isTraining, name='id2')
            y = self._op_ID_block(y, isTraining, name='id3')
        with tf.variable_scope('stage4', reuse=True):
            y = self._op_conv_block(y, isTraining, strides=2, name='conv')
            y = self._op_ID_block(y, isTraining, name='id1')
            y = self._op_ID_block(y, isTraining, name='id2')
            y = self._op_ID_block(y, isTraining, name='id3')
            y = self._op_ID_block(y, isTraining, name='id4')
            y = self._op_ID_block(y, isTraining, name='id5')
        with tf.variable_scope('stage5', reuse=True):
            y = self._op_conv_block(y, isTraining, strides=2, name='conv')
            y = self._op_ID_block(y, isTraining, name='id1')
            y = self._op_ID_block(y, isTraining, name='id2')
            
            y = tf.nn.conv2d(y, filter=tf.get_variable('W'), strides=[1, 1, 1, 1], padding='VALID')
            y = tf.reshape(y, shape=[-1, self.category])
            
            # y = tf.nn.avg_pool(y, ksize=[1, 2, 2, 1], strides=[
            #                    1, 1, 1, 1], padding='VALID')
            # y = tf.nn.conv2d(y, filter=tf.get_variable(
            #     'W'), strides=[1, 1, 1, 1], padding='SAME')
            # y = tf.reshape(y, shape=[-1, self.category])
        return y

    def _op_conv_block(self, X, isTraining, strides, name):
        with tf.variable_scope(name):
            with tf.variable_scope('a'):
                y = self._op_conv_bn_relu(X, isTraining, conv_strides=strides, conv_padding='VALID')
            with tf.variable_scope('b'):
                y = self._op_conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='SAME')
            with tf.variable_scope('c'):
                y = self._op_conv_bn(y, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('d'):
                y2 = self._op_conv_bn(X, isTraining, conv_strides=strides, conv_padding='VALID')
            y = y + y2
            y = tf.nn.relu(y)
        return y

    def _op_ID_block(self, X, isTraining, name):
        with tf.variable_scope(name):
            with tf.variable_scope('a'):
                y = self._op_conv_bn_relu(
                    X, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('b'):
                y = self._op_conv_bn_relu(
                    y, isTraining, conv_strides=1, conv_padding='SAME')
            with tf.variable_scope('c'):
                y = self._op_conv_bn(
                    y, isTraining, conv_strides=1, conv_padding='VALID')
            y = y + X
            y = tf.nn.relu(y)
        return y

    def _op_conv_bn_relu_pool(self, X, isTraining, conv_strides=1, conv_padding='SAME',
                              pool_size=2, pool_strides=2, pool_padding='SAME'):
        Z1 = tf.nn.conv2d(X, tf.get_variable('W'), strides=[
                          1, conv_strides, conv_strides, 1], padding=conv_padding, name='conv')
        if isTraining:
            M1, V1 = tf.nn.moments(
                Z1, axes=[0, 1, 2], shift=1e-8, keep_dims=True, name='moments')
            running_var_op = tf.assign(tf.get_variable(
                'var'), (self.momentum * V1 + (1 - self.momentum) * V1), name='running_var')
            self.update_op.append(running_var_op)
            running_mean_op = tf.assign(tf.get_variable(
                'mean'), (self.momentum * M1 + (1 - self.momentum) * M1), name='running_mean')
            self.update_op.append(running_mean_op)
        else:
            M1, V1 = tf.get_variable('mean'), tf.get_variable('var')
        scale1, offset1 = tf.get_variable('scale'), tf.get_variable('offset')
        B1 = tf.nn.batch_normalization(
            Z1, M1, V1, offset1, scale1, variance_epsilon=1e-8, name='bn')
        A1 = tf.nn.relu(B1, name='relu')
        P1 = tf.nn.max_pool(A1, ksize=[1, pool_size, pool_size, 1],
                            strides=[1, pool_strides, pool_strides, 1], padding=pool_padding, name='pool')
        return P1

    def _op_conv_bn_relu(self, X, isTraining, conv_strides=1, conv_padding='SAME'):
        Z1 = tf.nn.conv2d(X, tf.get_variable('W'), strides=[
                          1, conv_strides, conv_strides, 1], padding=conv_padding, name='conv')
        if isTraining:
            M1, V1 = tf.nn.moments(
                Z1, axes=[0, 1, 2], shift=1e-8, keep_dims=True, name='moments')
            running_var_op = tf.assign(tf.get_variable(
                'var'), (self.momentum * V1 + (1 - self.momentum) * V1), name='running_var')
            self.update_op.append(running_var_op)
            running_mean_op = tf.assign(tf.get_variable(
                'mean'), (self.momentum * M1 + (1 - self.momentum) * M1), name='running_mean')
            self.update_op.append(running_mean_op)
        else:
            M1, V1 = tf.get_variable('mean'), tf.get_variable('var')
        scale1, offset1 = tf.get_variable('scale'), tf.get_variable('offset')
        B1 = tf.nn.batch_normalization(
            Z1, M1, V1, offset1, scale1, variance_epsilon=1e-8, name='bn')
        A1 = tf.nn.relu(B1, name='relu')
        return A1

    def _op_conv_bn(self, X, isTraining, conv_strides=1, conv_padding='SAME'):
        Z1 = tf.nn.conv2d(X, tf.get_variable('W'), strides=[
                          1, conv_strides, conv_strides, 1], padding=conv_padding, name='conv')
        if isTraining:
            M1, V1 = tf.nn.moments(
                Z1, axes=[0, 1, 2], shift=1e-8, keep_dims=True, name='moments')
            running_var_op = tf.assign(tf.get_variable(
                'var'), (self.momentum * V1 + (1 - self.momentum) * V1), name='running_var')
            self.update_op.append(running_var_op)
            running_mean_op = tf.assign(tf.get_variable(
                'mean'), (self.momentum * M1 + (1 - self.momentum) * M1), name='running_mean')
            self.update_op.append(running_mean_op)
        else:
            M1, V1 = tf.get_variable('mean'), tf.get_variable('var')
        scale1, offset1 = tf.get_variable('scale'), tf.get_variable('offset')
        B1 = tf.nn.batch_normalization(
            Z1, M1, V1, offset1, scale1, variance_epsilon=1e-8, name='bn')
        return B1

    def _var_get_resnet_indentity_block_skip_3(self, name, filter_size, input_channels, filters_num=(64, 64, 64), isAddW2Loss=True):
        F1, F2, F3 = filters_num
        with tf.variable_scope(name):
            with tf.variable_scope('a'):
                self._var_get_conv_bn(
                    1, F1, input_channels, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('b'):
                self._var_get_conv_bn(
                    filter_size, F2, F1, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('c'):
                self._var_get_conv_bn(1, F3, F2, isAddW2Loss=isAddW2Loss)

    def _var_get_resnet_conv_block_skip_3(self, name, filter_size, input_channels, filters_num=(64, 64, 64), isAddW2Loss=True):
        F1, F2, F3 = filters_num
        with tf.variable_scope(name):
            with tf.variable_scope('a'):
                self._var_get_conv_bn(
                    1, F1, input_channels, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('b'):
                self._var_get_conv_bn(
                    filter_size, F2, F1, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('c'):
                self._var_get_conv_bn(1, F3, F2, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('d'):
                self._var_get_conv_bn(
                    1, F3, input_channels, isAddW2Loss=isAddW2Loss)

    def _var_get_conv_bn(self, filter_size=3, output_channels=64, input_channels=3,
                         W_stddev=0.001, isAddW2Loss=True):
        self.seed += 1
        tf.set_random_seed(self.seed)
        tf.get_variable('W', shape=[filter_size, filter_size, input_channels, output_channels],
                        dtype=self.dtype, initializer=tf.random_normal_initializer(mean=0.0, stddev=W_stddev))
        tf.get_variable('b', shape=[output_channels],
                        dtype=self.dtype, initializer=tf.zeros_initializer(dtype=self.dtype))
        tf.get_variable('scale', shape=[1, 1, 1, output_channels],
                        dtype=self.dtype, initializer=tf.ones_initializer(dtype=self.dtype))
        tf.get_variable('offset', shape=[1, 1, 1, output_channels],
                        dtype=self.dtype, initializer=tf.zeros_initializer(dtype=self.dtype))
        tf.get_variable('mean', shape=[1, 1, 1, output_channels],
                        dtype=self.dtype, initializer=tf.zeros_initializer(dtype=self.dtype), trainable=False)
        tf.get_variable('var', shape=[1, 1, 1, output_channels],
                        dtype=self.dtype, initializer=tf.ones_initializer(dtype=self.dtype), trainable=False)
        if isAddW2Loss:
            tf.get_variable_scope().reuse_variables()
            tf.add_to_collection(
                'loss_W', tf.reduce_sum(tf.get_variable('W')**2))

    def _var_get_conv(self, filter_size=3, output_channels=64, input_channels=3,
                      W_stddev=0.001, isAddW2Loss=True):
        self.seed += 2
        tf.set_random_seed(self.seed)
        tf.get_variable('W', shape=[filter_size, filter_size, input_channels, output_channels],
                        dtype=self.dtype, initializer=tf.random_normal_initializer(mean=0.0, stddev=W_stddev))
        tf.get_variable('b', shape=[output_channels],
                        dtype=self.dtype, initializer=tf.zeros_initializer(dtype=self.dtype))
        if isAddW2Loss:
            tf.get_variable_scope().reuse_variables()
            tf.add_to_collection(
                'loss_W', tf.reduce_sum(tf.get_variable('W')**2))
        
class GaussNet_Model(ResNet_Model):
    """GaussNet Model"""
    def __init__(self,
                 momentum=0.997,
                 batch_size=64,
                 seed=0,
                 category=10,
                 dtype=tf.float32,
                 X_shape=[None, 28, 28, 1], y_shape=[None],
                 start_reg=0,
                 starter_learning_rate=1e-2,
                 decay_steps=800,
                 decay_rate=0.5,
                 epchos=1):

        np.random.seed = seed
        self.dtype = dtype

        self.category = category
        self.momentum = momentum
        self.global_step = None
        self.learning_rate = None
        self.reg = tf.Variable(start_reg, trainable=False, name='reg')

        self.seed = seed
        self.batch_size = batch_size
        self.starter_learning_rate = starter_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.epchos = epchos
        
        self.epcho_now = 0

        self.update_op = []

        self.X = tf.placeholder(self.dtype, X_shape, 'X')
        self.y = tf.placeholder(self.dtype, y_shape, 'y')

        self.gauss_list = None

        self.op_y = None
        self.op_loss = None
        self.op_acc = None

        self.op_y_eval = None
        self.op_loss_eval = None
        self.op_acc_eval = None

        self.log_idex = []
        self.log_loss = []
        self.log_acc = []
        self.log_learning_rate = []

        self.log_idex_eval = []
        self.log_loss_eval = []
        self.log_acc_eval = []

        self.config = {}
        self.log = {
            'train_loss':{'x':[], 'y':[]},
            'train_acc':{'x':[], 'y':[]},
            'learning_rate':{'x':[], 'y':[]},
            'train_loss_epcho':{'x':[], 'y':[]},
            'valid_loss_epcho':{'x':[], 'y':[]},
            'train_acc_epcho':{'x':[], 'y':[]},
            'valid_acc_epcho':{'x':[], 'y':[]},
            'epcho':{'x':[], 'y':[]}
        }

    def gauss(self, x, y, mu1, mu2, sigma1, sigma2, ro):
        return 1/(2*np.pi*sigma1*sigma2*tf.sqrt(1-ro**2)) * tf.exp( -1/(2*(1-ro**2)) * ( ((x-mu1)/sigma1)**2 - 2*ro*(x-mu1)*(y-mu2)/(sigma1*sigma2+1e-3) + ((y-mu2)/(sigma2+1e-3))**2  ))

    def show_log(self, style='-o'):
        """show log"""
        plt.plot(self.log['train_loss']['x'], self.log['train_loss']['y'])
        plt.plot(self.log['train_loss_epcho']['x'], self.log['train_loss_epcho']['y'], style)
        plt.plot(self.log['valid_loss_epcho']['x'], self.log['valid_loss_epcho']['y'], style)
        plt.legend(['train loss step', 'train loss epcho', 'valid loss epcho'])
        plt.title('Loss')
        plt.show()
        plt.plot(self.log['train_acc']['x'], self.log['train_acc']['y'])
        plt.plot(self.log['train_acc_epcho']['x'], self.log['train_acc_epcho']['y'], style)
        plt.plot(self.log['valid_acc_epcho']['x'], self.log['valid_acc_epcho']['y'], style)
        plt.legend(['train acc step', 'train acc epcho', 'valid acc epcho'])
        plt.title('Accuracy')
        plt.show()

    def _log_append(self, name, x, y):
        self.log[name]['x'].append(x)
        self.log[name]['y'].append(y)

    def log_append(self, sess, train_loss, train_acc):
        x = sess.run(self.global_step)
        self._log_append('train_loss', x, train_loss)
        self._log_append('train_acc', x, train_acc)
        self._log_append('learning_rate', x, self.learning_rate)

    def log_append_to_epcho(self, sess, train_loss, train_acc, valid_loss, valid_acc):
        x = sess.run(self.global_step)
        self._log_append('train_loss_epcho', x, train_loss)
        self._log_append('train_acc_epcho', x, train_acc)
        self._log_append('valid_loss_epcho', x, valid_loss)
        self._log_append('valid_acc_epcho', x, valid_acc)
        self._log_append('epcho', x, self.epcho_now)

    def get_variable(self):


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
                self._var_get_conv_bn(filter_size=3, output_channels=32, input_channels=1, W_stddev = 0.001, isAddW2Loss=True)
            with tf.variable_scope('conv2'):
                self._var_get_conv_bn(filter_size=3, output_channels=64, input_channels=32, W_stddev = 0.001)
            with tf.variable_scope('conv3'):
                self._var_get_conv_bn(filter_size=3, output_channels=128, input_channels=64, W_stddev = 0.01)
            with tf.variable_scope('conv4'):
                self._var_get_conv_bn(filter_size=1, output_channels=256, input_channels=128, W_stddev = 0.1)
            with tf.variable_scope('conv5'):
                self._var_get_conv(filter_size=1, output_channels=5, input_channels=256, W_stddev = 0.1)
        # H, W, C, F
        with tf.variable_scope('forward'):
            with tf.variable_scope('conv0'):
                self._var_get_conv_bn(filter_size=3, output_channels=32, input_channels=1, W_stddev = 0.001, isAddW2Loss=True)
            with tf.variable_scope('conv_neck'):
                self._var_get_conv_bn(filter_size=3, output_channels=1, input_channels=32, W_stddev = 0.001, isAddW2Loss=True)            
            with tf.variable_scope('conv1'):
                self._var_get_conv_bn(filter_size=3, output_channels=32, input_channels=1, W_stddev = 0.001, isAddW2Loss=True)
            with tf.variable_scope('conv2'):
                self._var_get_conv_bn(filter_size=3, output_channels=64, input_channels=32, W_stddev = 0.001)
            with tf.variable_scope('conv3'):
                self._var_get_conv_bn(filter_size=3, output_channels=64, input_channels=64, W_stddev = 0.01)
            with tf.variable_scope('conv4'):
                self._var_get_conv_bn(filter_size=1, output_channels=256, input_channels=64, W_stddev = 0.1)
            with tf.variable_scope('conv5'):
                self._var_get_conv(filter_size=1, output_channels=10, input_channels=256, W_stddev = 0.1)

    def see_gauss(self, sess, pic, isTraining=True):
        return sess.run(self.gauss_list, feed_dict={self.X:pic})

    def _op_forward(self, isTraining):
        with tf.variable_scope('calculate_mu1'):
            with tf.variable_scope('conv1', reuse=True):
                y = self._op_conv_bn_relu_pool(self.X, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv2', reuse=True):
                y = self._op_conv_bn_relu_pool(y, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv3', reuse=True):
                y = self._op_conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
                y = tf.nn.avg_pool(y, ksize=[1,4,4,1], strides=[1,1,1,1], padding='VALID')
            with tf.variable_scope('conv4', reuse=True):
                y = self._op_conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('conv5', reuse=True):
                y = tf.nn.conv2d(y, filter=tf.get_variable('W'), strides=[1,1,1,1], padding='VALID')
                mu1 = tf.reshape(tf.nn.tanh(y[:,:,:,0]), shape=[-1,1,1,1])+1e-6
                mu2 = tf.reshape(tf.nn.tanh(y[:,:,:,1]), shape=[-1,1,1,1])+1e-6
                sigma1 = tf.reshape(tf.nn.sigmoid(y[:,:,:,2]), shape=[-1,1,1,1])+1e-6
                sigma2 = tf.reshape(tf.nn.sigmoid(y[:,:,:,3]), shape=[-1,1,1,1])+1e-6
                ro = tf.reshape(tf.nn.tanh(y[:,:,:,4]), shape=[-1,1,1,1])/1.001                
                self.gauss_list = [mu1, mu2, sigma1, sigma2, ro]
        with tf.variable_scope('gauss', reuse=True):
            p = self.gauss(tf.get_variable('gauss_X'), tf.get_variable('gauss_Y'), mu1, mu2, sigma1, sigma2, ro)
        with tf.variable_scope('forward', reuse=True):
            with tf.variable_scope('conv0', reuse=True):
                y = self._op_conv_bn_relu(self.X, isTraining, conv_strides=1, conv_padding='SAME')
            with tf.variable_scope('conv_neck', reuse=True):
                y = self._op_conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='SAME')
                y = y * p
            with tf.variable_scope('conv1', reuse=True):
                y = self._op_conv_bn_relu_pool(y, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv2', reuse=True):
                y = self._op_conv_bn_relu_pool(y, isTraining, conv_strides=1, conv_padding='VALID', pool_size=2, pool_strides=2, pool_padding='SAME')
            with tf.variable_scope('conv3', reuse=True):
                y = self._op_conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
                y = tf.nn.avg_pool(y, ksize=[1,4,4,1], strides=[1,1,1,1], padding='VALID')
            with tf.variable_scope('conv4', reuse=True):
                y = self._op_conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('conv5', reuse=True):
                y = tf.nn.conv2d(y, filter=tf.get_variable('W'), strides=[1,1,1,1], padding='VALID')
                y = tf.reshape(y, shape=[-1,10])        
        return y

    def get_graph(self):
        """Get Graph"""
        self.op_y = self._op_forward(isTraining=True)
        loss = tf.losses.softmax_cross_entropy(tf.one_hot(tf.cast(self.y, tf.int32), self.category), self.op_y)
        self.op_loss = tf.reduce_mean(loss) + self.reg * tf.add_n(tf.get_collection('loss_W'))
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.op_y, 1), tf.int32), tf.cast(self.y, tf.int32))
        self.op_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.op_y_eval = self._op_forward(isTraining=False)
        loss = tf.losses.softmax_cross_entropy(tf.one_hot(tf.cast(self.y, tf.int32), self.category), self.op_y_eval)
        self.op_loss_eval = tf.reduce_mean(loss) + self.reg * tf.add_n(tf.get_collection('loss_W'))
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.op_y_eval, 1), tf.int32), tf.cast(self.y, tf.int32))
        self.op_acc_eval = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                        self.decay_steps, self.decay_rate, staircase=True, name='learning_rate')
        AdamOptimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step = AdamOptimizer.minimize(self.op_loss, global_step=self.global_step)
        self.optimizer = [self.update_op, train_step]

    def evaluate(self, sess, data):
        bs = self.batch_size
        X_eval, y_eval = data
        m = np.shape(X_eval)[0]
        loss_real_list, accuracy_real_list = [], []
        tm = m // bs + 1
        for i in range(tm):
            X_batch = X_eval[i * bs:i * bs + bs, :]
            y_batch = y_eval[i * bs:i * bs + bs]
            feed_dict = {self.X: X_batch, self.y: y_batch}
            loss_real_batch, accuracy_real_batch = sess.run([self.op_loss_eval, self.op_acc_eval], feed_dict=feed_dict)
            loss_real_list.append(loss_real_batch)
            accuracy_real_list.append(accuracy_real_batch)
        loss_real, accuracy_real = np.sum(loss_real_list)/tm, np.sum(accuracy_real_list)/tm
        return loss_real, accuracy_real

    def train(self, sess, data):
        bs = self.batch_size
        X_eval, y_eval = data
        m = np.shape(X_eval)[0]
        loss_real_list, accuracy_real_list = [], []
        tm = m // bs
        for i in range(tm):
            X_batch = X_eval[i*bs:i*bs+bs, :]
            y_batch = y_eval[i*bs:i*bs+bs]
            feed_dict = {self.X: X_batch, self.y: y_batch}
            loss_real_batch, accuracy_real_batch, _ = sess.run(
                [self.op_loss, self.op_acc, self.optimizer], feed_dict=feed_dict)
            loss_real_list.append(loss_real_batch)
            accuracy_real_list.append(accuracy_real_batch)
            self.log_append(sess, loss_real_batch, accuracy_real_batch)
        loss_real, accuracy_real = np.sum(loss_real_list)/tm, np.sum(accuracy_real_list)/tm
        return loss_real, accuracy_real

    def run(self, sess, data):
        """X_y_holder: (X, y)
        data = (X_train, y_train, X_valid, y_valid, X_test, y_test)
        """
        X_train, y_train, X_valid, y_valid, X_test, y_test = data
        data_train = (X_train, y_train)
        data_valid = (X_valid, y_valid)
        data_test = (X_test, y_test)

        loss_real_train, accuracy_real_train = self.evaluate(sess, data_train)
        loss_real_valid, accuracy_real_valid = self.evaluate(sess, data_valid)
        self.log_append_to_epcho(sess, loss_real_train, accuracy_real_train, loss_real_valid, accuracy_real_valid)
        self.log_append(sess, loss_real_train, accuracy_real_train)

        print('epoch %d Loss: train %0.5f, valid %0.5f, acc: train %0.5f, valid %0.5f' % (
            0, loss_real_train, loss_real_valid, accuracy_real_train, accuracy_real_valid))

        for epcho in range(1,self.epchos+1):
            np.random.seed = self.seed + 1
            X_train, y_train = data_train
            idx = np.random.permutation(X_train.shape[0])
            X_train_tmp = X_train[idx, :]
            y_train_tmp = y_train[idx]
            data_train = (X_train_tmp, y_train_tmp)
            loss_real_train, accuracy_real_train = self.train(sess, data_train)
            loss_real_valid, accuracy_real_valid = self.evaluate(sess, data_valid)
            self.log_append_to_epcho(sess, loss_real_train, accuracy_real_train, loss_real_valid, accuracy_real_valid)

            with open('min_loss1.txt', 'r+') as fp:
                line = fp.readline()
                tmp_acc = float(line)
                if accuracy_real_valid > tmp_acc:
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, "model.ckpt")
                    print("Model saved in file: %s" % save_path)
                    fp.seek(0, 0)
                    fp.truncate()
                    fp.write(str(accuracy_real_valid))

            print('epoch %d Loss: train %0.5f, valid %0.5f, acc: train %0.5f, valid %0.5f' % (
                epcho, loss_real_train, loss_real_valid, accuracy_real_train, accuracy_real_valid))
            print('global step %d, learning rate %f' %
                  (sess.run(self.global_step), sess.run(self.learning_rate)))

    def _op_conv_block(self, X, isTraining, strides, name):
        with tf.variable_scope(name):
            with tf.variable_scope('a'):
                y = self._op_conv_bn_relu(X, isTraining, conv_strides=strides, conv_padding='VALID')
            with tf.variable_scope('b'):
                y = self._op_conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='SAME')
            with tf.variable_scope('c'):
                y = self._op_conv_bn(y, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('d'):
                y2 = self._op_conv_bn(X, isTraining, conv_strides=strides, conv_padding='VALID')
            y = y + y2
            y = tf.nn.relu(y)
        return y

    def _op_ID_block(self, X, isTraining, name):
        with tf.variable_scope(name):
            with tf.variable_scope('a'):
                y = self._op_conv_bn_relu(
                    X, isTraining, conv_strides=1, conv_padding='VALID')
            with tf.variable_scope('b'):
                y = self._op_conv_bn_relu(
                    y, isTraining, conv_strides=1, conv_padding='SAME')
            with tf.variable_scope('c'):
                y = self._op_conv_bn(
                    y, isTraining, conv_strides=1, conv_padding='VALID')
            y = y + X
            y = tf.nn.relu(y)
        return y

    def _op_conv_bn_relu_pool(self, X, isTraining, conv_strides=1, conv_padding='SAME',
                              pool_size=2, pool_strides=2, pool_padding='SAME'):
        Z1 = tf.nn.conv2d(X, tf.get_variable('W'), strides=[
                          1, conv_strides, conv_strides, 1], padding=conv_padding, name='conv')
        if isTraining:
            M1, V1 = tf.nn.moments(
                Z1, axes=[0, 1, 2], shift=1e-8, keep_dims=True, name='moments')
            running_var_op = tf.assign(tf.get_variable(
                'var'), (self.momentum * V1 + (1 - self.momentum) * V1), name='running_var')
            self.update_op.append(running_var_op)
            running_mean_op = tf.assign(tf.get_variable(
                'mean'), (self.momentum * M1 + (1 - self.momentum) * M1), name='running_mean')
            self.update_op.append(running_mean_op)
        else:
            M1, V1 = tf.get_variable('mean'), tf.get_variable('var')
        scale1, offset1 = tf.get_variable('scale'), tf.get_variable('offset')
        B1 = tf.nn.batch_normalization(
            Z1, M1, V1, offset1, scale1, variance_epsilon=1e-8, name='bn')
        A1 = tf.nn.relu(B1, name='relu')
        P1 = tf.nn.max_pool(A1, ksize=[1, pool_size, pool_size, 1],
                            strides=[1, pool_strides, pool_strides, 1], padding=pool_padding, name='pool')
        return P1

    def _op_conv_bn_relu(self, X, isTraining, conv_strides=1, conv_padding='SAME'):
        Z1 = tf.nn.conv2d(X, tf.get_variable('W'), strides=[
                          1, conv_strides, conv_strides, 1], padding=conv_padding, name='conv')
        if isTraining:
            M1, V1 = tf.nn.moments(
                Z1, axes=[0, 1, 2], shift=1e-8, keep_dims=True, name='moments')
            running_var_op = tf.assign(tf.get_variable(
                'var'), (self.momentum * V1 + (1 - self.momentum) * V1), name='running_var')
            self.update_op.append(running_var_op)
            running_mean_op = tf.assign(tf.get_variable(
                'mean'), (self.momentum * M1 + (1 - self.momentum) * M1), name='running_mean')
            self.update_op.append(running_mean_op)
        else:
            M1, V1 = tf.get_variable('mean'), tf.get_variable('var')
        scale1, offset1 = tf.get_variable('scale'), tf.get_variable('offset')
        B1 = tf.nn.batch_normalization(
            Z1, M1, V1, offset1, scale1, variance_epsilon=1e-8, name='bn')
        A1 = tf.nn.relu(B1, name='relu')
        return A1

    def _op_conv_bn(self, X, isTraining, conv_strides=1, conv_padding='SAME'):
        Z1 = tf.nn.conv2d(X, tf.get_variable('W'), strides=[
                          1, conv_strides, conv_strides, 1], padding=conv_padding, name='conv')
        if isTraining:
            M1, V1 = tf.nn.moments(
                Z1, axes=[0, 1, 2], shift=1e-8, keep_dims=True, name='moments')
            running_var_op = tf.assign(tf.get_variable(
                'var'), (self.momentum * V1 + (1 - self.momentum) * V1), name='running_var')
            self.update_op.append(running_var_op)
            running_mean_op = tf.assign(tf.get_variable(
                'mean'), (self.momentum * M1 + (1 - self.momentum) * M1), name='running_mean')
            self.update_op.append(running_mean_op)
        else:
            M1, V1 = tf.get_variable('mean'), tf.get_variable('var')
        scale1, offset1 = tf.get_variable('scale'), tf.get_variable('offset')
        B1 = tf.nn.batch_normalization(
            Z1, M1, V1, offset1, scale1, variance_epsilon=1e-8, name='bn')
        return B1

    def _var_get_resnet_indentity_block_skip_3(self, name, filter_size, input_channels, filters_num=(64, 64, 64), isAddW2Loss=True):
        F1, F2, F3 = filters_num
        with tf.variable_scope(name):
            with tf.variable_scope('a'):
                self._var_get_conv_bn(
                    1, F1, input_channels, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('b'):
                self._var_get_conv_bn(
                    filter_size, F2, F1, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('c'):
                self._var_get_conv_bn(1, F3, F2, isAddW2Loss=isAddW2Loss)

    def _var_get_resnet_conv_block_skip_3(self, name, filter_size, input_channels, filters_num=(64, 64, 64), isAddW2Loss=True):
        F1, F2, F3 = filters_num
        with tf.variable_scope(name):
            with tf.variable_scope('a'):
                self._var_get_conv_bn(
                    1, F1, input_channels, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('b'):
                self._var_get_conv_bn(
                    filter_size, F2, F1, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('c'):
                self._var_get_conv_bn(1, F3, F2, isAddW2Loss=isAddW2Loss)
            with tf.variable_scope('d'):
                self._var_get_conv_bn(
                    1, F3, input_channels, isAddW2Loss=isAddW2Loss)

    def _var_get_conv_bn(self, filter_size=3, output_channels=64, input_channels=3,
                         W_stddev=0.001, isAddW2Loss=True):
        self.seed += 1
        tf.set_random_seed(self.seed)
        tf.get_variable('W', shape=[filter_size, filter_size, input_channels, output_channels],
                        dtype=self.dtype, initializer=tf.random_normal_initializer(mean=0.0, stddev=W_stddev))
        tf.get_variable('b', shape=[output_channels],
                        dtype=self.dtype, initializer=tf.zeros_initializer(dtype=self.dtype))
        tf.get_variable('scale', shape=[1, 1, 1, output_channels],
                        dtype=self.dtype, initializer=tf.ones_initializer(dtype=self.dtype))
        tf.get_variable('offset', shape=[1, 1, 1, output_channels],
                        dtype=self.dtype, initializer=tf.zeros_initializer(dtype=self.dtype))
        tf.get_variable('mean', shape=[1, 1, 1, output_channels],
                        dtype=self.dtype, initializer=tf.zeros_initializer(dtype=self.dtype), trainable=False)
        tf.get_variable('var', shape=[1, 1, 1, output_channels],
                        dtype=self.dtype, initializer=tf.ones_initializer(dtype=self.dtype), trainable=False)
        if isAddW2Loss:
            tf.get_variable_scope().reuse_variables()
            tf.add_to_collection(
                'loss_W', tf.reduce_sum(tf.get_variable('W')**2))

    def _var_get_conv(self, filter_size=3, output_channels=64, input_channels=3,
                      W_stddev=0.001, isAddW2Loss=True):
        self.seed += 2
        tf.set_random_seed(self.seed)
        tf.get_variable('W', shape=[filter_size, filter_size, input_channels, output_channels],
                        dtype=self.dtype, initializer=tf.random_normal_initializer(mean=0.0, stddev=W_stddev))
        tf.get_variable('b', shape=[output_channels],
                        dtype=self.dtype, initializer=tf.zeros_initializer(dtype=self.dtype))
        if isAddW2Loss:
            tf.get_variable_scope().reuse_variables()
            tf.add_to_collection(
                'loss_W', tf.reduce_sum(tf.get_variable('W')**2))
        
    
if __name__ == '__main__':
    mod = ResNet_Model()
