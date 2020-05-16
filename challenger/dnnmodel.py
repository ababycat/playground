#!usr/bin/env python3

"""dense neural network
"""

import math
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

class dnn_model:
    def __init__(self):
        """init
        """
        self.struct = {
            'layer0':{
                'num': 108
            },
            'layer1':{
                'num': 50,
                'random_seed':1,
                'stddev':0.01
            },
            'layer2':{
                'num': 50,
                'random_seed':2,
                'stddev':0.01
            },
            'layer3':{
                'num': 50,
                'random_seed':3,
                'stddev':0.001
            },
            'layer4':{
                'num': 20,
                'random_seed':4,
                'stddev':0.001
            },
            'layer5':{
                'num': 20,
                'random_seed':5,
                'stddev':0.001
            },
            'layer6':{
                'num': 20,
                'random_seed':3,
                'stddev':0.001
            },
            'layer7':{
                'num': 20,
                'random_seed':4,
                'stddev':0.01
            },
            'layer8':{
                'num': 20,
                'random_seed':3,
                'stddev':0.01
            },
            'layer9':{
                'num': 20,
                'random_seed':4,
                'stddev':0.001
            },
            'layer10':{
                'num': 20,
                'random_seed':3,
                'stddev':0.01
            },
            'layer11':{
                'num': 20,
                'random_seed':4,
                'stddev':0.001
            },
            'layer12':{
                'num': 20,
                'random_seed':3,
                'stddev':0.01
            },
            'layer13':{
                'num': 20,
                'random_seed':4,
                'stddev':0.001
            },
            'layer14':{
                'num': 2,
                'random_seed':5,
                'stddev':0.01
            }
        }
        self.layer_num = len(self.struct.items())-1
        self.layer = {}
        self.hyper_params = {}

        self.train_X, self.train_y = None, None
        self.valid_y, self.valid_y = None, None
        self.test_X, self.test_y = None, None
        
        self.loss_log = {
            'train':[],
            'valid':[],
            'test':[],
            'train_losses':[]
        }

    def _get_layer_param(self, num_n_minus_l, num_n, stddev=0.001, dtype=tf.float32, seed=0):
        """get layer param
        """
        tf.set_random_seed(seed)
        W = tf.Variable(tf.random_normal(shape=[num_n_minus_l, num_n],
                            mean=0.0,
                            stddev=stddev,
                            dtype=dtype), dtype=dtype, name='layer_W')
        b = tf.Variable(tf.zeros(num_n), dtype=dtype, name='layer_b')
        return W, b

    def _get_layer_bn_param(self, num_n_minus_l, num_n, stddev=0.001, dtype=tf.float32, seed=1):
        """get layer bn param
        """
        tf.set_random_seed(seed)
        W, b = self._get_layer_param(num_n_minus_l, num_n, stddev, dtype)
        bn_scale = tf.Variable(tf.ones(num_n), dtype=dtype, name='layer_bn_scale')
        bn_offset = tf.Variable(tf.zeros(num_n), dtype=dtype, name='layer_bn_offset')
        bn_running_mean = tf.Variable(tf.zeros(num_n), dtype=dtype, name='bn_running_mean')
        bn_running_var = tf.Variable(tf.zeros(num_n), dtype=dtype, name='bn_running_var')
        return W, b, bn_scale, bn_offset, bn_running_mean, bn_running_var

    def bacth_normalization(self, layer, l, offset, scale, momentum = 0.9, isTraining=True, eps=1e-8):
        if isTraining:
            mu, var = tf.nn.moments(layer, axes=1, keep_dims=True)
            self.layer['layer'+str(l)]['bn_running_mean'] = momentum*mu + (1-momentum)*self.layer['layer'+str(l)]['bn_running_mean']
            self.layer['layer'+str(l)]['bn_running_var'] = momentum*var + (1-momentum)*self.layer['layer'+str(l)]['bn_running_var']
        else:
            mu, var = self.layer['bn_running_mean'], self.layer['bn_running_var']
        return tf.nn.batch_normalization(layer, mu, var, offset, scale, eps)
        
    def _get_param(self):
        """get params
        """
        for l in range(1, self.layer_num):
            self.layer['layer'+str(l)] = {}
            with tf.variable_scope('layer'+str(l)):
                self.layer['layer'+str(l)]['W'], \
                self.layer['layer'+str(l)]['b'], \
                self.layer['layer'+str(l)]['bn_scale'], \
                self.layer['layer'+str(l)]['bn_offset'], \
                self.layer['layer'+str(l)]['bn_running_mean'], \
                self.layer['layer'+str(l)]['bn_running_var'] \
                                                    = self._get_layer_bn_param(self.struct['layer'+str(l-1)]['num'], 
                                                                                self.struct['layer'+str(l)]['num'],
                                                                                seed=self.struct['layer'+str(l)]['random_seed'],
                                                                                stddev=self.struct['layer'+str(l)]['stddev'])
        l = self.layer_num
        self.layer['layer'+str(l)] = {}
        with tf.variable_scope('layer'+str(l)):
            self.layer['layer'+str(l)]['W'], \
            self.layer['layer'+str(l)]['b'] = self._get_layer_param(self.struct['layer'+str(l-1)]['num'],
                                                                self.struct['layer'+str(l)]['num'],
                                                                seed = self.struct['layer'+str(l)]['random_seed'])

    def forward(self, X, isTraining=True):
        last_layer = X
        for l in range(1, self.layer_num):
            layer = tf.add(tf.matmul(last_layer, self.layer['layer'+str(l)]['W']), self.layer['layer'+str(l)]['b'],name='l1')
            bn = self.bacth_normalization(layer, l, self.layer['layer'+str(l)]['bn_offset'], 
                                                self.layer['layer'+str(l)]['bn_scale'], isTraining)
            last_layer = tf.nn.relu(bn)
        l = self.layer_num
        Z = tf.add(tf.matmul(last_layer, self.layer['layer'+str(l)]['W']), self.layer['layer'+str(l)]['b'], name='l2')
        return Z

    def loss(self, y, z, lamb):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(tf.cast(y, tf.int32),2), logits=z))
        for l in range(1, self.layer_num+1):
            loss += lamb*tf.reduce_sum(self.layer['layer'+str(l)]['W']**2)
        return loss
    
    def accuarcy(self, y, z):
        correct_prediction = tf.equal(tf.argmax(z,1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def train(self, sess, train_X, train_y, val_X, val_y, test_X, test_y, epochs = 10, starter_learning_rate = 0.001, decay_steps=6000, decay_rate=0.6, batch_size = 64, regular = 0.001, dtype=tf.float32):
        m, n = train_X.shape
        # clear param first
        self._get_param()
        
        X = tf.placeholder(dtype=dtype, shape=[None, n])
        y = tf.placeholder(dtype=dtype, shape=[None])
        z = self.forward(X, isTraining=True)
        loss = self.loss(y, z, regular)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                decay_steps, decay_rate, staircase=True, name='learning_rate')
        AdamOptimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step = AdamOptimizer.minimize(loss, global_step=self.global_step)

        loss_test_op = self.loss(y, self.forward(X, isTraining=False), regular)

        variables = [loss, train_step]

        sess.run(tf.global_variables_initializer())
    
        index = np.arange(m)
        
        for i in range(epochs):
            for j in range(int(np.ceil(m/batch_size))):
                idx = index[j*batch_size:j*batch_size+batch_size]
                batch_X = train_X[idx, :]
                batch_y = train_y[idx]
                feed_dict={X:batch_X, y:batch_y}
                loss_batch, _ = sess.run(variables, feed_dict=feed_dict)
                self.loss_log['train_losses'].append(loss_batch)
            loss_train = np.sum(self.loss_log['train_losses'])/len(self.loss_log['train_losses'])
            self.loss_log['train'].append(loss_train)
            
            feed_dict={X:val_X, y:val_y}
            loss_val = sess.run(loss_test_op, feed_dict=feed_dict)
            self.loss_log['valid'].append(loss_val)

            feed_dict={X:test_X, y:test_y}
            loss_test = sess.run(loss_test_op, feed_dict=feed_dict)
            self.loss_log['test'].append(loss_test)
            
            print('epcho %d:train loss:%0.6f, valid loss:%0.6f, test loss%0.6f'%(i, loss_train, loss_val, loss_test))
            print('global step %d, learning rate %f'%(sess.run(self.global_step), sess.run(self.learning_rate)))

    def predict(self, X):
        z = self.forward(X, isTraining=False)
        y_pred = tf.nn.softmax(z)
        return y_pred

    def plot_log(self):
        plt.plot(self.loss_log['train'], '-o')
        plt.plot(self.loss_log['valid'], '-o')
        plt.plot(self.loss_log['test'], '-o')
        plt.legend(['train', 'valid', 'test'])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
