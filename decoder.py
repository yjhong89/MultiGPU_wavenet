import numpy as np
import tensorflow as tf
import time, os, collections
import data_loader
from wavenet_model import Wavenet_Model
from ops import *

class DECODER():
    def __init__(self,args, sess):
        self.args = args
        self.sess = sess

        self.global_step = tf.Variable(0, trainable=False)

        # Get test data
        with tf.device('/cpu:0'):
            print('\tLoading test data')
            self.args.num_gpu = 1
            test_wave, test_label, test_seq_len = data_loader.get_batches(data_category='test', shuffle=self.args.shuffle, batch_size=self.args.batch_size, num_gpu=self.args.num_gpu, num_threads=2)
       
        self.test_net = Wavenet_Model(self.args, test_wave, test_label, test_seq_len, self.global_step, name='test')
        self.test_net.build_model()
        # To load checkpoint
        self.saver = tf.train.Saver()
        self.decode()
    
    
    def decode(self):
        self.sess.run(tf.global_variables_initializer())

        if self.load():
            print('Load checkpoint')
        else:
            raise Exception('No ckpt!')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            char_prob, decoded, ler_ = self.sess.run([self.test_net.probs, self.test_net.dcd, self.test_net.ler])

            print('Label Error rate : %3.4f' % ler_)
    
            decoded_original = reverse_sparse_tensor(decoded[0][0])
            
            # [batch size, number of steps, number of classes]
            char_prob = np.asarray(char_prob)
            # Get greedy index
            high_index = np.argmax(char_prob, axis=2)
            
            str_decoded = list()
            for i in range(len(decoded_original)):
            	str_decoded.append(''.join([chr(x) for x in np.asarray(decoded_original[i]) + SpeechLoader.FIRST_INDEX]))
            	if self.args.num_classes == 30:
            		# 27:Space, 28:Apstr, 29:<EOS>, last class:blank
            		str_decoded[i] = str_decoded[i].replace(chr(ord('z')+4), "")
            		str_decoded[i] = str_decoded[i].replace(chr(ord('z')+3), '.')
            		str_decoded[i] = str_decoded[i].replace(chr(ord('z')+2), "'")
            		str_decoded[i] = str_decoded[i].replace(chr(ord('z')+1), ' ')
            	elif self.args.num_classes == 29:
            		# 27:Space, 28:Apstr, last class:blank
            		str_decoded[i] = str_decoded[i].replace(chr(ord('z')+3), "")
            		str_decoded[i] = str_decoded[i].replace(chr(ord('z')+2), "'")
            		str_decoded[i] = str_decoded[i].replace(chr(ord('z')+1), ' ')
            	print(str_decoded[i])

        except KeyboardInterrupt:
            print('Keyboard')
        finally:
            coord.request_stop()
            coord.join(threads)    				
    
    @property
    def model_dir(self):
        return '{}blocks_{}layers_{}width_{}'.format(self.args.num_blocks, self.args.num_wavenet_layers, self.args.filter_width, self.args.dilated_activation)

    def load(self):
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.init_epoch = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            self.init_epoch = 0
            return False   
