#!usr/bin/env python3
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

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
    
    def get_varibale(self, dtype=tf.float32):
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
        with tf.variable_scope('conv4', reuse=True):
            y = self.conv_bn_relu(y, isTraining, conv_strides=1, conv_padding='VALID')
            print(y)
        with tf.variable_scope('conv5', reuse=True):
            y = tf.nn.conv2d(y, filter=tf.get_variable('W'), strides=[1,1,1,1], padding='VALID')
            y = tf.reshape(y, shape=[-1,10])
        return y
    
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

        sess.run(tf.global_variables_initializer())

        X_y_holder = (X, y)
        self.run(sess, data, X_y_holder, loss_train, accuracy_train, loss_val_test, accuracy_val_test, optimizer, batch_size=batch_size, epchos=epchos)        

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
