#!/usr/bin/python
# -*- coding : utf-8 -*-

import numpy as np
import tensorflow as tf
import argparse, time, os, sys
import collections
import ops 
import data_loader
from wavenet_model import Wavenet_Model


class Multi_GPU_train():
    def __init__(self, args, sess):
        self.args = args
        self.sess = sess
        '''
            tf.Graph is a collection of operation
            This method should be used if you want to create multiple graphs in the sampe process
            The default graph is a property of the current thread. If you create a new thread, and wish to use the default graph
            in that thread, you must explicitly add a 'with as_default() in that thread function
        '''
#        with tf.Graph().as_default():    
        self.global_step = tf.Variable(0, trainable=False)

        # Get waves and labels
        import multiprocessing
        num_threads = multiprocessing.cpu_count() // self.args.num_gpu
        print('Load data with %d threads' % num_threads)
        with tf.device('/cpu:0'):
            print('\tLoading training data')
            with tf.variable_scope('train_data'):
                train_wave, train_label, train_seq_len = data_loader.get_batches(data_category='train', shuffle=self.args.shuffle, batch_size=self.args.batch_size, num_gpu=self.args.num_gpu, num_threads=num_threads)
            print('\tLoading valid data')
            with tf.variable_scope('valid_data'):
                test_wave, test_label, test_seq_len = data_loader.get_batches(data_category='valid', shuffle=self.args.shuffle, batch_size=self.args.batch_size, num_gpu=self.args.num_gpu, num_threads=num_threads)
                
        # Build model
        self.train_net = Wavenet_Model(self.args, train_wave, train_label, train_seq_len, global_step, name='train')
        self.train_net.build_model()                
        self.train_net.train_optimizer()
        self.train_summary_op = tf.summary.merge_all()
        self.valid_net = Wavenet_Model(selfargs, test_wave, test_label, test_seq_len, global_step, name='valid', reuse=True)
        self.valid_net.build_model() 
        
        # Checkpoint with maximum checkpoints to keep 5
        self.saver = tf.train.Saver(max_to_keep=5)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        
        if self.load():
            print('Load checkpoint')
        else:
            print('No checkpoint')

        summary_writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        
        tf.train.start_queue_runners(sess=self.sess)
        
        for epoch in xrange(self.init_epoch, self.args.num_epoch):
            start_time = time.time()
            # Train
            _, loss_, ler_, train_summary = self.sess.run([self.train_net.train_op, self.train_net.losses, self.train_net.ler, self.train_summary_op])

            if epoch % self.args.valid_interval == 0:
                # Valid
                best_valid_ler = 1000
                best_valid_loss = 1000
                valid_loss = 0
                valid_ler = 0
                # Conduct validation several types for different composition of batch
                for valid_iter in range(self.args.valid_iteration):
                    valid_loss_, valid_ler_ = self.sess.run([self.valid_net.losses, self.valid_net.ler])
                    valid_loss += valid_loss_
                    valid_ler += valid_ler_
                valid_loss /= self.args.valid_iteration
                valid_ler /= self.args.valid_iteration
    
                # Tensor log for validation values
                valid_summary = tf.Summary()
                valid_summary.value.add(tag='valid/loss', simple_value=value_loss)
                valid_summary.value.add(tag='valid/ler', simple_value=value_ler)
                valid_summary.value.add(tag='valid/best_valid_ler', simple_value=best_valid_ler)
                summary_writer.add_summary(valid_summary, epoch)
                self.write_log(epoch, loss_, ler_, valid_loss, valid_ler, start_time)
                summary_writer.add_summary(train_summary, epoch)
                summary_writer.flush()
    
                if best_valid_ler > valid_ler:
                    best_valid_ler = min(best_valid_ler, valid_ler)
                    best_valid_loss = min(best_valid_loss, valid_loss)
                    # Save only when validation improved
                    self.save(global_step=epoch)
            

    @property
    def model_dir(self):
        return '{}blocks_{}layers_{}width_{}'.format(self.args.num_blocks, self.args.num_wavenet_layers, sef.args.filter_width, self.args.dilated_activation)

    def save(self, global_step):
        model_name='WAVENET_MG'
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        print('Checkpoint saved')   

    def load(self):
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.init_epoch = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            self.init_epoch = 0
            return False   

    def write_log(self, epoch, loss, ler, valid_loss, valid_ler, start_time):
        print('Write logs..')
        log_path = os.path.join(self.args.log_dir, self.model_dir+'.csv')
        if not os.path.exists(log_path):
       	    self.log_file = open(log_path, 'w')
            self.log_file.write('Epoch,\tavg_loss,\tavg_ler,\tvalid_loss,\tvalid_ler,\ttime\n')
        else:
            self.log_file = open(log_path, 'a')
        
        self.log_file.write('%d,\t%3.4f,\t%3.4f,\t%3.4f,\t%3.4f,\t%3.4f sec\n' % (epoch, loss, ler, valid_loss, valid_ler, time.time()-start_time))
        print('At epoch %d, train loss:%3.3f, train ler:%3.3f, valid loss:%3.3f, valid_ler:%3.3f' % (epoch, loss, ler, valid_loss, valid_ler))
        self.log_file.flush()


