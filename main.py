#!/usr/bin/python
# -*- coding : utf-8 -*-

import numpy as np
import tensorflow as tf
import argparse, time, os, sys
from ops import *
from wavenet_model import Wavenet_Model
from train import Multi_GPU_train 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='../train', help='data directory containing audio clip')
    parser.add_argument('--test_dir', type=str, default='../test', help='data directory containing audio clip and transcription')
    #parser.add_argument('--valid_data_dir', type=str, default='./validation')
    parser.add_argument('--files_dir', type=str, default='./files')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='To restore variables and model')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_gpu', type=int, default=2)
    parser.add_argument('--num_epoch', type=int, default=100000)
    parser.add_argument('--valid_interval', type=int, default=1000)
    parser.add_argument('--valid_iteration', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--is_train', type=str2bool, default='t')
    parser.add_argument('--layer_norm', type=str2bool, default='t')
    parser.add_argument('--init_from', type=str2bool, default='n', help='Continue training from saved model')
    parser.add_argument('--shuffle', type=str2bool, default='t') 
    parser.add_argument('--num_features', type=int, default=39)
    parser.add_argument('--num_classes', type=int, default=29, help='All lowercase letter, space, apstr, eos, blank : last class is always reserved for blank')
    parser.add_argument('--seq_length', type=int, default=200, help='number of steps')
    parser.add_argument('--mode', type=str2bool, default='n', help='No for ctc, Yes for clm')
    parser.add_argument('--alpha', type=float, default=2.0, help='language model weight')
    parser.add_argument('--beta', type=float, default=1.5, help='insertion bonus')
    parser.add_argument('--beam_width', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--filter_width', type=int, default=9)
    parser.add_argument('--skip_filter_width', type=int, default=3)
    parser.add_argument('--num_wavenet_layers', type=int, default=10)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--causal', type=str2bool, default='n')
    parser.add_argument('--dilated_activation', type=str, default='gated_linear', choices=['gated_linear', 'gated_tanh'])
    
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(args.checkpoint_dir):
    	os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.files_dir):
    	os.mkdir(args.files_dir)
    if not os.path.exists(args.log_dir):
    	os.mkdir(args.log_dir)
    
    run_config = tf.ConfigProto()
    # GPU fraction to be allocated
    #run_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.args.gpu_fraction)
    run_config.log_device_placement=False
    run_config.gpu_options.allow_growth=True
    run_config.allow_soft_placement=True
    
    with tf.Session(config=run_config) as sess:
    	if args.is_train:
            print('Training')
            multi_gpu_sr = Multi_GPU_train(args, sess) 
            multi_gpu_sr.train()    
    	else:
            from decoder import DECODER
            print('Decoding')	
            decoding = DECODER(args, sess)

    
def str2bool(v):
	if v.lower() in ('yes', 'y', 'true', 't', 1):
		return True
	elif v.lower() in ('no', 'n', 'false', 'f', 0):
		return False



if __name__ == '__main__':
 	main()

