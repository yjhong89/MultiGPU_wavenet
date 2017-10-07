import numpy as np
import tensorflow as tf
from ops import *
import time

class Wavenet_Model():
    def __init__(self, args, wave_batch_list, label_batch_list, seq_len_list, global_step, name=None, reuse=False):
        self.args = args
        self.reuse = reuse
        self.name = name
        self.wave_batch_list = wave_batch_list
        self.label_batch_list = label_batch_list
        self.seq_len_list = seq_len_list
        self.global_step = global_step

    def build_tower(self, waves, label, seq_len):
        print('Building model')
        # Do not need to make placeholders
        skip = 0
        '''
        	Construct of a stack of dilated causal convolutional layers
        '''
        # First non-causal convolution to inputs to expand feature dimension
        h = conv1d(waves, self.args.num_hidden, filter_width=self.args.filter_width, name='conv_in', normalization=self.args.layer_norm, activation=tf.nn.tanh)
        # As many as number of blocks, block means one total dilated convolution layers
        for blocks in range(self.args.num_blocks):
        	# Construction of dilation
        	for dilated in range(self.args.num_wavenet_layers):
        		# [1,2,4,8,16..]
        		rate = 2**dilated 
        		h, s = res_block(h, self.args.num_hidden, rate, self.args.causal, self.args.filter_width, normalization=self.args.layer_norm, activation=self.args.dilated_activation, name='{}block_{}layer'.format(blocks+1, dilated+1))
        		skip += s
        # Make skip connections
        with tf.variable_scope('postprocessing'):
        	# 1*1 convolution
        	skip = conv1d(tf.nn.relu(skip), self.args.num_hidden, filter_width=self.args.skip_filter_width, activation=tf.nn.relu, normalization=self.args.layer_norm, name='conv_out1')
        	hidden = conv1d(skip, self.args.num_hidden, filter_width=self.args.skip_filter_width, activation=tf.nn.relu, normalization=self.args.layer_norm, name='conv_out2')
        	logits = conv1d(hidden, self.args.num_classes, filter_width=1, activation=None, normalization=self.args.layer_norm, name='conv_out3')

        probability = tf.nn.softmax(logits)
        
        # To calculate ctc, consider timemajor
        logits_reshaped = tf.transpose(logits, [1,0,2])
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels=label, inputs=logits_reshaped, sequence_length=seq_len))
        decoded, _ = tf.nn.ctc_greedy_decoder(logits_reshaped, seq_len)	
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), label))

        return logits, probibility, loss, ler

    def build_model(self):
        # Build towers for each GPU
        self.logits_list = list()
        self.prob_list = list()
        self.loss_list = list()
        self.ler_list = list()

        for i in range(self.args.num_gpu):
            with tf.device('/gpu:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                # tf.name_scope is ignored by tf.get_variable
                with tf.name_scope('tower_%d' % i) as scope:
                    print('Building %s' % scope)
                    # If gpu index is not 0, reuse variable
                    if self.reuse or i > 0:
                        # Call reuse_variables()
                        tf.get_variable_scope.reuse_variables()
                        
                    logits, prob, loss, ler = self.build_tower(self.wave_batch_list[i], self.label_batch_list[i], self.seq_len_list[i])
                    self.logits_list.append(logits)
                    self.prob_list.append(prob)
                    self.loss_list.append(loss)
                    self.ler_list.append(ler)

        # Merge losses and error rates of all GPUs
        with tf.device('/cpu:0'):
            self.logits = tf.concat(self.logits_list, axis=0, name='logit')
            self.probs = tf.concat(self.prob_list, axis=0, name='prob')
            self.losses = tf.reduce_mean(self.loss_list, name='loss')
            self.ler = tf.reduce_mean(self.ler_list, name='ler')
            tf.summary.scalar(self.name+'ctc_loss', self.losses)
            tf.summary.scalar(self.name+'ler', self.ler)
        
    def train_optimizer(self):
        optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
        
        # Contain variables and corresponding gradient w.r.t loss
        self.grad_vars = list()

        # Compute gradients for each GPU
        for i in range(self.args.num_gpu):
            with tf.device('/gpu:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i) as scope:
                    print('Compute gradients of %s' % scope)
                    if self.reuse or i >0:
                        tf.get_variable_scope.reuse_variables()
                
                    trainable_vr = tf.trainable_variables()
                    # compute_gradients outputs list of tuples(grad,var)
                    grad_var = optimizer.compute_gradients(self.loss_list[i], trainable_vr)
                
                    self.grad_vars.append(grad_var)

        # Averaging gradients
        print('Averaging gradients')
        with tf.device('/cpu:0'):
            grads_vars = self.average_gradients(self.grad_vars)
            # When use tf.contrib.layers.layer_norm(batch_norm), update_ops are placed in tf.GraphKeys.UPDATE_OPS so they need to be added as a dependency to the train_op
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.apply_gradients(grads_vars, global_step=self.global_step)

            # '*' to represent op list
            #self.train_op = tf.group(*(update_ops+[apply_grad_ops]))

                        
    def average_gradients(self, grad_and_var):
        '''
            Calculate the average gradient for each shared variable across all towers
            synchronous version

            Args:
                List of lists of tuples(grad, var)
            Returns:
                List of pairs of (grad, var) which gradients have been averaged over all tower
        '''
        average_grad = list()
        '''
            zip(*grad_and_vars) returns..
                ((grad0_gpu0, var0_gpu0), ...(grad0_gpuN, var0_gpuN)),
                ...
                ((gradM_gpu0, varM_gpu0),...(gradM_gpuN, varM_gpuN)) 
        '''
        for g_v in zip(*grad_and_var):
            '''
                g_v returns..
                    ((grad0_gpu0, var0_gpu0), ..(grad0_gpuN, var0_gpuN))
            '''
            
            # If no gradient for variable, exclude it
            if g_v[0][0] is None:
                continue

            grads = list()
            for gradient, _ in g_v:
                # Add 0 axis to represent tower
                expanded_gradient = tf.expand_dims(gradient, 0)
                # Append on tower dimension
                grads.append(expanded_gradient)

            # Concatenate and average over tower dimension
            grad = tf.reduce_mean(tf.concat(grads, axis=0))
            
            # Since variables are shared across towers, so return only the first tower`s variable
            var = g_v[0][1]
            g_and_v = (grad, var)
            average_grad.append(g_and_v)
        
        return average_grad

